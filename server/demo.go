// Demo-video harness endpoints. These are NOT part of the reconstruction API —
// they exist so the dedicated storyboard page (web/demo.html) can be rendered to
// an offline, deterministic MP4: the page steps a fixed frame schedule, reads
// each frame back with gl.readPixels, and POSTs the PNGs here; the server then
// shells out to ffmpeg (host tool, same recipe as ~/c/LocalVQE/demo) to encode.
//
//   GET  /demo-assets/...           static files from -demo-dir (manifest.json, *.splat, source images)
//   POST /api/demo/frame?session=S&k=N   body = image/png  -> <work>/S/f_%05d.png
//   POST /api/demo/encode?session=S&fps=F -> runs ffmpeg     -> {video: "/api/demo/video/S"}
//   GET  /api/demo/video/{session}  -> the encoded MP4
//
// The page-streams-to-server design is deliberate: there is no node/puppeteer on
// this host, and it lets the same page render on the real GPU (open it in a
// normal browser on the 5070) or headless via swiftshader (CI), unchanged.
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// sessionRe-equivalent: keep session/name tokens to a safe charset so they can be
// used as path components without traversal. Anything else is rejected.
func safeToken(s string) (string, bool) {
	if s == "" || len(s) > 64 {
		return "", false
	}
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_'
		if !ok {
			return "", false
		}
	}
	return s, true
}

func (s *server) framesDir(session string) string {
	return filepath.Join(s.workDir, "frames", session)
}

// POST /api/demo/clear?session=S  — wipe a session's frames before a fresh render,
// so a shorter render can't inherit a previous run's higher-numbered (stale) frames.
func (s *server) handleDemoClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	session, ok := safeToken(r.URL.Query().Get("session"))
	if !ok {
		http.Error(w, "bad session", http.StatusBadRequest)
		return
	}
	if err := os.RemoveAll(s.framesDir(session)); err != nil {
		http.Error(w, "clear: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// POST /api/demo/frame?session=S&k=N  — body is a PNG; written as f_%05d.png.
func (s *server) handleDemoFrame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	session, ok := safeToken(r.URL.Query().Get("session"))
	if !ok {
		http.Error(w, "bad session", http.StatusBadRequest)
		return
	}
	k, err := strconv.Atoi(r.URL.Query().Get("k"))
	if err != nil || k < 0 || k > 1_000_000 {
		http.Error(w, "bad frame index", http.StatusBadRequest)
		return
	}
	dir := s.framesDir(session)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		http.Error(w, "mkdir: "+err.Error(), http.StatusInternalServerError)
		return
	}
	data, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	path := filepath.Join(dir, fmt.Sprintf("f_%05d.png", k))
	if err := os.WriteFile(path, data, 0o644); err != nil {
		http.Error(w, "write frame: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// POST /api/demo/encode?session=S&fps=F  — assemble frames into an MP4.
func (s *server) handleDemoEncode(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	session, ok := safeToken(r.URL.Query().Get("session"))
	if !ok {
		http.Error(w, "bad session", http.StatusBadRequest)
		return
	}
	fps := r.URL.Query().Get("fps")
	if _, err := strconv.Atoi(fps); err != nil {
		fps = "30"
	}
	dir := s.framesDir(session)
	if _, err := os.Stat(filepath.Join(dir, "f_00000.png")); err != nil {
		http.Error(w, "no frames for session (post frames first)", http.StatusBadRequest)
		return
	}
	if err := os.MkdirAll(filepath.Join(s.workDir, "video"), 0o755); err != nil {
		http.Error(w, "mkdir: "+err.Error(), http.StatusInternalServerError)
		return
	}
	out := filepath.Join(s.workDir, "video", session+".mp4")
	// Same encode recipe as the LocalVQE harness: H.264, yuv420p, crf 18, faststart.
	cmd := exec.Command("ffmpeg", "-y", "-loglevel", "error",
		"-framerate", fps, "-i", filepath.Join(dir, "f_%05d.png"),
		"-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
		"-movflags", "+faststart", out)
	if combined, err := cmd.CombinedOutput(); err != nil {
		http.Error(w, "ffmpeg: "+err.Error()+": "+string(combined), http.StatusInternalServerError)
		return
	}
	fi, _ := os.Stat(out)
	respondJSON(w, map[string]any{
		"video":   "/api/demo/video/" + session,
		"path":    out,
		"bytes":   fi.Size(),
		"session": session,
	})
}

// GET /api/demo/video/{session}
func (s *server) handleDemoVideo(w http.ResponseWriter, r *http.Request) {
	session, ok := safeToken(strings.TrimPrefix(r.URL.Path, "/api/demo/video/"))
	if !ok {
		http.Error(w, "bad session", http.StatusBadRequest)
		return
	}
	out := filepath.Join(s.workDir, "video", session+".mp4")
	f, err := os.Open(out)
	if err != nil {
		http.Error(w, "no such video", http.StatusNotFound)
		return
	}
	defer f.Close()
	fi, _ := f.Stat()
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Length", strconv.FormatInt(fi.Size(), 10))
	w.Header().Set("Cache-Control", "no-store")
	io.Copy(w, f)
}
