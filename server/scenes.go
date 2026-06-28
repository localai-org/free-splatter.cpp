// scenes.go — the accumulating-reconstruction side of the web app: list the baked
// "growing world" scenes, and generate a new one from an uploaded video in-process
// (reuse the already-loaded GPU engine, no shell-out, no second model load).
//
//   GET  /api/scenes                 -> [{name,label,steps,thumb,source}, ...]
//   POST /api/scene/from-video        multipart "video" + "name" -> {job}
//   GET  /api/scene/status/{job}      -> {state,total,done,kept,scene,error}
//   GET  /scenes-assets/<name>/...    static scene files (manifest.json, *.splat, view_*.jpg)
//
// A scene is a subdir of scenesDir whose manifest.json has a top-level "steps"
// array (acc_2.splat, acc_3.splat, ... + a fused step) — the format produced by
// scripts/make_accumulate_demo.sh and by handleSceneFromVideo here. The viewer
// (server/web/accumulate.html) plays the steps as the reconstruction grows.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// --- manifest format (shared with server/web/accumulate.html + make_accumulate_demo.sh) ---

type manifestStep struct {
	Splat  string   `json:"splat"`
	Images []string `json:"images"`
	N      int      `json:"n"`
	Label  string   `json:"label,omitempty"`
}
type sceneManifest struct {
	Steps []manifestStep `json:"steps"`
}

// --- listing -------------------------------------------------------------------

type sceneInfo struct {
	Name   string `json:"name"`
	Label  string `json:"label"`
	Steps  int    `json:"steps"`
	Thumb  string `json:"thumb"` // /scenes-assets/<name>/<first image>, for the picker
	Source string `json:"source"`
}

// scanScenes lists the accumulating-reconstruction scenes under scenesDir: each
// immediate subdir with a manifest.json that has a non-empty "steps" array. Cheap
// enough to run per request, so externally-baked and uploaded scenes appear with
// no restart.
func (s *server) scanScenes() []sceneInfo {
	if s.scenesDir == "" {
		return nil
	}
	ents, err := os.ReadDir(s.scenesDir)
	if err != nil {
		return nil
	}
	out := []sceneInfo{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		raw, err := os.ReadFile(filepath.Join(s.scenesDir, name, "manifest.json"))
		if err != nil {
			continue
		}
		var m sceneManifest
		if json.Unmarshal(raw, &m) != nil || len(m.Steps) == 0 {
			continue // storyboard manifests ("stations") and malformed dirs are skipped
		}
		thumb := ""
		if len(m.Steps[0].Images) > 0 {
			thumb = "/scenes-assets/" + name + "/" + m.Steps[0].Images[0]
		}
		label := strings.ReplaceAll(name, "-", " ")
		out = append(out, sceneInfo{
			Name: name, Label: label, Steps: len(m.Steps), Thumb: thumb, Source: "baked",
		})
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Name < out[b].Name })
	return out
}

func (s *server) handleScenes(w http.ResponseWriter, r *http.Request) {
	respondJSON(w, map[string]any{"scenes": s.scanScenes()})
}

// --- video -> scene (async) ----------------------------------------------------

type sceneJob struct {
	State string `json:"state"` // "running" | "done" | "error"
	Total int    `json:"total"` // candidate frames
	Done  int    `json:"done"`  // candidates processed
	Kept  int    `json:"kept"`  // frames folded in so far
	Scene string `json:"scene"` // slug, set on done
	Err   string `json:"error,omitempty"`
}

func (s *server) putJob(id string, mutate func(*sceneJob)) {
	s.jobsMu.Lock()
	defer s.jobsMu.Unlock()
	j := s.jobs[id]
	if j == nil {
		j = &sceneJob{}
		s.jobs[id] = j
	}
	mutate(j)
}

func (s *server) handleSceneStatus(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/scene/status/")
	s.jobsMu.Lock()
	j := s.jobs[id]
	var cp sceneJob
	if j != nil {
		cp = *j
	}
	s.jobsMu.Unlock()
	if j == nil {
		http.Error(w, "no such job", http.StatusNotFound)
		return
	}
	respondJSON(w, cp)
}

// handleSceneFromVideo accepts a clip, returns a job id immediately, and bakes the
// gated accumulating scene in a background goroutine (max one at a time).
func (s *server) handleSceneFromVideo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if s.scenesDir == "" {
		http.Error(w, "no scenes dir configured (-scenes-dir / -demo-dir)", http.StatusServiceUnavailable)
		return
	}
	if err := r.ParseMultipartForm(256 << 20); err != nil {
		http.Error(w, "bad multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	name, ok := safeToken(r.FormValue("name"))
	if !ok {
		http.Error(w, "bad or missing scene name (a-z0-9-_ only)", http.StatusBadRequest)
		return
	}
	minParallax := float32(8)
	if v, err := strconv.ParseFloat(r.FormValue("min_parallax"), 32); err == nil {
		minParallax = float32(v)
	}
	maxFrames := 24
	if v, err := strconv.Atoi(r.FormValue("max_frames")); err == nil && v >= 2 && v <= 200 {
		maxFrames = v
	}
	file, _, err := r.FormFile("video")
	if err != nil {
		http.Error(w, `no video (expected multipart field "video")`, http.StatusBadRequest)
		return
	}
	defer file.Close()

	// one bake at a time — a bake monopolizes the single engine for many GPU-seconds.
	select {
	case s.bakeSem <- struct{}{}:
	default:
		http.Error(w, "a scene bake is already running; try again shortly", http.StatusTooManyRequests)
		return
	}

	jobDir := filepath.Join(s.workDir, "scene", name)
	if err := os.MkdirAll(jobDir, 0o755); err != nil {
		<-s.bakeSem
		http.Error(w, "mkdir: "+err.Error(), http.StatusInternalServerError)
		return
	}
	inPath := filepath.Join(jobDir, "input")
	in, err := os.Create(inPath)
	if err != nil {
		<-s.bakeSem
		http.Error(w, "save upload: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(in, file); err != nil {
		in.Close()
		<-s.bakeSem
		http.Error(w, "save upload: "+err.Error(), http.StatusInternalServerError)
		return
	}
	in.Close()

	jobID := name // one bake at a time + scene name as slug -> stable status URL
	s.putJob(jobID, func(j *sceneJob) { *j = sceneJob{State: "running"} })

	go func() {
		defer func() { <-s.bakeSem }()
		scene, err := s.bakeSceneFromVideo(name, inPath, jobDir, minParallax, maxFrames)
		s.putJob(jobID, func(j *sceneJob) {
			if err != nil {
				j.State, j.Err = "error", err.Error()
				return
			}
			j.State, j.Scene = "done", scene
		})
	}()

	respondJSON(w, map[string]any{"job": jobID, "name": name})
}

// bakeSceneFromVideo: sample frames -> gated in-process accumulate -> scene dir.
func (s *server) bakeSceneFromVideo(name, inPath, jobDir string, minParallax float32, maxFrames int) (string, error) {
	eng := s.models[s.order[0]] // scene model is the default/first
	if eng == nil {
		return "", fmt.Errorf("no scene model loaded")
	}
	sz := int(eng.Geo.imageWidth)
	gc := int(eng.Geo.gaussianChannels)

	// 1) decode all frames, then even-stride down to <= maxFrames (time-lapse clips
	//    are short; decoding all avoids frame-count guessing — mirrors bake-vids.sh).
	frameDir := filepath.Join(jobDir, "frames")
	os.RemoveAll(frameDir)
	if err := os.MkdirAll(frameDir, 0o755); err != nil {
		return "", err
	}
	if err := runFFmpeg("-i", inPath, filepath.Join(frameDir, "all%04d.png")); err != nil {
		return "", fmt.Errorf("frame extraction: %w", err)
	}
	all, _ := filepath.Glob(filepath.Join(frameDir, "all*.png"))
	sort.Strings(all)
	if len(all) < 2 {
		return "", fmt.Errorf("video decoded to %d frames (need >= 2)", len(all))
	}
	stride := (len(all) + maxFrames - 1) / maxFrames
	if stride < 1 {
		stride = 1
	}
	var pngs []string
	for i := 0; i < len(all); i += stride {
		pngs = append(pngs, all[i])
	}

	// 2) preprocess sampled frames to model input (CHW [0,1]).
	frames := make([][]float32, len(pngs))
	for i, p := range pngs {
		data, err := os.ReadFile(p)
		if err != nil {
			return "", err
		}
		px, err := preprocessImage(data, sz)
		if err != nil {
			return "", fmt.Errorf("preprocess %s: %w", filepath.Base(p), err)
		}
		frames[i] = px
	}
	s.putJob(name, func(j *sceneJob) { j.Total = len(frames) - 1 })

	// 3) gated in-process accumulate (mirrors free_splatter-cli.cpp keyframe loop).
	acc := eng.lib.accNew(int32(sz), int32(sz), s.opts.opacThr)
	if acc == 0 {
		return "", fmt.Errorf("accumulator alloc failed")
	}
	defer eng.lib.accFree(acc)

	sceneDir := filepath.Join(s.scenesDir, name)
	if err := os.MkdirAll(sceneDir, 0o755); err != nil {
		return "", err
	}
	// frame 0 is the anchor; thumbnail it as view_0.jpg.
	if err := thumbnail(pngs[0], filepath.Join(sceneDir, "view_0.jpg")); err != nil {
		return "", err
	}

	keptIdx := []int{0}
	lastKept := 0
	for j := 1; j < len(frames); j++ {
		pair := make([]float32, 0, 2*len(frames[0]))
		pair = append(pair, frames[lastKept]...)
		pair = append(pair, frames[j]...)
		g, err := eng.Run(pair, 2)
		if err != nil {
			return "", fmt.Errorf("inference pair (%d,%d): %w", lastKept, j, err)
		}
		if minParallax > 0 {
			if px, ok := eng.lib.pairParallax(g, int32(sz), int32(sz), int32(gc), s.opts.opacThr); ok && px.TriAngleDeg < minParallax {
				s.putJob(name, func(jb *sceneJob) { jb.Done = j })
				continue // re-anchor: lastKept unchanged, try the next candidate
			}
		}
		if eng.lib.accAddPairSlice(acc, g, int32(gc)) != 0 {
			return "", fmt.Errorf("add_pair (%d,%d) failed", lastKept, j)
		}
		nframes := int(eng.lib.accFrameCnt(acc))
		cloud := eng.lib.accCloudCopy(acc)
		splat := encodeCloudSplat(cloud, s.opts.maxSplats, 1.0)
		if err := os.WriteFile(filepath.Join(sceneDir, fmt.Sprintf("acc_%d.splat", nframes)), splat, 0o644); err != nil {
			return "", err
		}
		keptIdx = append(keptIdx, j)
		if err := thumbnail(pngs[j], filepath.Join(sceneDir, fmt.Sprintf("view_%d.jpg", len(keptIdx)-1))); err != nil {
			return "", err
		}
		lastKept = j
		s.putJob(name, func(jb *sceneJob) { jb.Done, jb.Kept = j, len(keptIdx) })
	}
	if len(keptIdx) < 2 {
		return "", fmt.Errorf("only %d frame kept (min-parallax %.0f too high for this clip?)", len(keptIdx), minParallax)
	}

	// 4) consensus-fused surface (best-frame, k=2, voxel 0.02 — same as the CLI bake).
	fused := encodeCloudSplat(eng.lib.accFuseCopy(acc, 0.02, 2, 2), s.opts.maxSplats, 1.0)
	if err := os.WriteFile(filepath.Join(sceneDir, "acc_fused.splat"), fused, 0o644); err != nil {
		return "", err
	}

	// 5) manifest: one step per kept frame (acc_2..acc_N) + the fused step.
	if err := writeSceneManifest(sceneDir, len(keptIdx)); err != nil {
		return "", err
	}
	return name, nil
}

// writeSceneManifest emits the steps manifest from `nkept` kept frames, referencing
// view_0.jpg .. view_<k-1>.jpg per step — identical shape to make_accumulate_demo.sh.
func writeSceneManifest(dir string, nkept int) error {
	var m sceneManifest
	imgs := func(k int) []string {
		out := make([]string, k)
		for j := 0; j < k; j++ {
			out[j] = fmt.Sprintf("view_%d.jpg", j)
		}
		return out
	}
	for k := 2; k <= nkept; k++ {
		m.Steps = append(m.Steps, manifestStep{Splat: fmt.Sprintf("acc_%d.splat", k), Images: imgs(k), N: k})
	}
	m.Steps = append(m.Steps, manifestStep{
		Splat: "acc_fused.splat", Images: imgs(nkept), N: nkept,
		Label: "consensus-fused — best-frame, parallax-gated",
	})
	raw, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "manifest.json"), raw, 0o644)
}

// thumbnail center-crops src to a square and scales to 360px, mirroring the engine
// preprocessing (so the filmstrip shows exactly the reconstructed view).
func thumbnail(src, dst string) error {
	return runFFmpeg("-i", src, "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=360:360", dst)
}

func runFFmpeg(args ...string) error {
	full := append([]string{"-y", "-loglevel", "error"}, args...)
	if out, err := exec.Command("ffmpeg", full...).CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
