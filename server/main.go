// free-splatter web app — a Go HTTP server that binds the free_splatter C API
// via purego (Vulkan by default), turns dropped photos into a gaussian splat,
// and serves a WebGL frontend (embedded) that renders it with a build-up reveal.
// One or more checkpoints (scene / object) can be loaded and switched per request.
//
//   cd server && go run . [-models scene=A.gguf,object=B.gguf] [-device vulkan] [-addr :8080]
//
// REST API:
//   GET  /api/models         -> [{name,label,hint,views,size}, ...]
//   POST /api/reconstruct     multipart "images" (>=1) + optional "model" -> {id,model,n_views,n_splats,size,seconds}
//   GET  /api/splat/{id}      -> the .splat bytes (importance-sorted, 32B/splat)
//   GET  /                    -> embedded frontend (web/)
package main

import (
	"embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

//go:embed web
var webFS embed.FS

const maxViews = 8 // guard against accidental OOM from a giant upload

// Per-model presentation + the view count the checkpoint was trained for.
type modelMeta struct {
	Label string
	Hint  string
	Views int
}

var modelInfo = map[string]modelMeta{
	"scene":       {"Scene", "2 photos of a scene from different angles — they must overlap.", 2},
	"object":      {"Object", "3–4 photos around a single object; a plain background works best.", 4},
	"object-2dgs": {"Object (2DGS)", "3–4 photos around a single object.", 4},
}

type server struct {
	models      map[string]*Engine
	order       []string // model names in load order; order[0] is the default
	opts        options
	bgRemoveCmd string // external matting command ("" = feature disabled)

	mu      sync.Mutex
	results map[string][]byte
	order2  []string
	counter int64
}

type options struct {
	maxSplats int
	opacThr   float32
}

func (s *server) store(b []byte) string {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.counter++
	id := strconv.FormatInt(s.counter, 36)
	s.results[id] = b
	s.order2 = append(s.order2, id)
	for len(s.order2) > 16 { // keep memory bounded: only the most recent results
		delete(s.results, s.order2[0])
		s.order2 = s.order2[1:]
	}
	return id
}

func (s *server) get(id string) []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.results[id]
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	type item struct {
		Name  string `json:"name"`
		Label string `json:"label"`
		Hint  string `json:"hint"`
		Views int    `json:"views"`
		Size  int    `json:"size"`
	}
	out := make([]item, 0, len(s.order))
	for _, name := range s.order {
		eng := s.models[name]
		meta := modelInfo[name]
		label := meta.Label
		if label == "" {
			label = name
		}
		out = append(out, item{name, label, meta.Hint, meta.Views, int(eng.Geo.imageWidth)})
	}
	resp := struct {
		Models   []item `json:"models"`
		BgRemove bool   `json:"bg_remove"`
	}{out, s.bgRemoveCmd != ""}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleReconstruct(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "bad multipart form: "+err.Error(), http.StatusBadRequest)
		return
	}
	modelName := r.FormValue("model")
	if modelName == "" {
		modelName = s.order[0]
	}
	eng := s.models[modelName]
	if eng == nil {
		http.Error(w, "unknown model: "+modelName, http.StatusBadRequest)
		return
	}
	files := r.MultipartForm.File["images"]
	if len(files) == 0 {
		http.Error(w, `no images (expected multipart field "images")`, http.StatusBadRequest)
		return
	}
	if len(files) > maxViews {
		http.Error(w, "too many views (max "+strconv.Itoa(maxViews)+")", http.StatusBadRequest)
		return
	}

	removeBg := r.FormValue("remove_bg") != "" && s.bgRemoveCmd != ""
	size := int(eng.Geo.imageWidth)

	uploads := make([][]byte, 0, len(files))
	for _, fh := range files {
		f, err := fh.Open()
		if err != nil {
			http.Error(w, "open upload: "+err.Error(), http.StatusBadRequest)
			return
		}
		data, err := io.ReadAll(io.LimitReader(f, 64<<20))
		f.Close()
		if err != nil {
			http.Error(w, "read upload: "+err.Error(), http.StatusBadRequest)
			return
		}
		uploads = append(uploads, data)
	}
	nViews := len(files)

	t0 := time.Now()
	buf := make([]float32, 0, len(uploads)*3*size*size)
	if removeBg {
		imgs, err := removeBackgrounds(s.bgRemoveCmd, uploads)
		if err != nil {
			http.Error(w, "background removal: "+err.Error(), http.StatusInternalServerError)
			return
		}
		for i, im := range imgs {
			px, err := preprocessDecoded(im, size)
			if err != nil {
				http.Error(w, fmt.Sprintf("preprocess view %d: %v", i, err), http.StatusBadRequest)
				return
			}
			buf = append(buf, px...)
		}
	} else {
		for i, data := range uploads {
			px, err := preprocessImage(data, size)
			if err != nil {
				http.Error(w, fmt.Sprintf("decode view %d: %v", i, err), http.StatusBadRequest)
				return
			}
			buf = append(buf, px...)
		}
	}

	g, err := eng.Run(buf, nViews)
	if err != nil {
		http.Error(w, "inference: "+err.Error(), http.StatusInternalServerError)
		return
	}
	splat, m := encodeSplat(g, s.opts.opacThr, s.opts.maxSplats)
	id := s.store(splat)
	tag := modelName
	if removeBg {
		tag += "+nobg"
	}
	log.Printf("reconstruct[%s]: %d views -> %d splats in %s", tag, nViews, m, time.Since(t0).Round(time.Millisecond))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": id, "model": modelName, "n_views": nViews, "n_splats": m, "size": size,
		"remove_bg": removeBg, "seconds": time.Since(t0).Seconds(),
	})
}

func (s *server) handleSplat(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/splat/")
	b := s.get(id)
	if b == nil {
		http.Error(w, "no such splat", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Length", strconv.Itoa(len(b)))
	w.Header().Set("Cache-Control", "no-store")
	w.Write(b)
}

// parseModels turns "scene=a.gguf,object=b.gguf" into ordered (name, absPath).
func parseModels(spec string) (names []string, paths []string) {
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		name, path, ok := strings.Cut(part, "=")
		if !ok { // bare path -> infer name from filename
			path = part
			name = "scene"
		}
		abs, _ := filepath.Abs(strings.TrimSpace(path))
		names = append(names, strings.TrimSpace(name))
		paths = append(paths, abs)
	}
	return
}

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	lib := flag.String("lib", "../build/vulkan/libfree_splatter.so", "path to libfree_splatter.so")
	models := flag.String("models", "scene=../.cache/freesplatter-scene-f16.gguf",
		"comma list of name=path GGUFs to load (first is the default), e.g. scene=a.gguf,object=b.gguf")
	device := flag.String("device", "vulkan", "compute device: cpu | vulkan | vulkan:N")
	maxSplats := flag.Int("max-splats", 300000, "cap on splats returned (0 = all)")
	opacThr := flag.Float64("opacity-threshold", 5e-3, "prune gaussians with opacity <= this")
	bgCmd := flag.String("bgremove-cmd", "",
		"external background-removal command for the object path ({in}/{out} are batch dirs), e.g. 'rembg p -m u2netp {in} {out}'")
	flag.Parse()

	libAbs, _ := filepath.Abs(*lib)
	log.Printf("opening %s", libAbs)
	l, err := OpenLib(libAbs)
	if err != nil {
		log.Fatalf("library: %v", err)
	}
	defer l.Close()

	names, paths := parseModels(*models)
	if len(names) == 0 {
		log.Fatal("no models specified (see -models)")
	}
	s := &server{
		models:      map[string]*Engine{},
		opts:        options{maxSplats: *maxSplats, opacThr: float32(*opacThr)},
		bgRemoveCmd: *bgCmd,
		results:     map[string][]byte{},
	}
	if *bgCmd != "" {
		log.Printf("background removal enabled: %s", *bgCmd)
	}
	for i, name := range names {
		log.Printf("loading model %q from %s on %s", name, paths[i], *device)
		eng, err := l.Load(paths[i], *device)
		if err != nil {
			log.Fatalf("load %q: %v", name, err)
		}
		defer eng.Close()
		s.models[name] = eng
		s.order = append(s.order, name)
		log.Printf("  ready: %dx%d, gaussian_channels=%d", eng.Geo.imageWidth, eng.Geo.imageHeight, eng.Geo.gaussianChannels)
	}

	sub, err := fs.Sub(webFS, "web")
	if err != nil {
		log.Fatal(err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models", s.handleModels)
	mux.HandleFunc("/api/reconstruct", s.handleReconstruct)
	mux.HandleFunc("/api/splat/", s.handleSplat)
	mux.Handle("/", http.FileServer(http.FS(sub)))

	log.Printf("serving %d model(s) %v on http://localhost%s", len(s.order), s.order, *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}
