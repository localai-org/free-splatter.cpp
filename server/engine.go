// engine.go — purego bindings to libfree_splatter (the flat C API declared in
// include/free_splatter.h). No cgo: the shared library is dlopen'd once (Lib) and
// each model is loaded as its own context (Engine), so several checkpoints can be
// served from one library. An Engine's context is NOT thread-safe (one ggml
// backend + one compute graph), so Run is serialized behind a per-Engine mutex.
package main

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// geometry mirrors free_splatter_geometry (four int32, same order).
type geometry struct {
	inChannels       int32
	imageHeight      int32
	imageWidth       int32
	gaussianChannels int32
}

// point mirrors free_splatter_point exactly: 14 float32 + 1 int32 = 60 bytes, all
// 4-byte aligned (no padding). Field ORDER is load-bearing across the C ABI.
type point struct {
	X, Y, Z, R, G, B, Opacity, Sx, Sy, Sz, Qw, Qx, Qy, Qz float32
	Frame                                                 int32
}

// parallax mirrors free_splatter_parallax: 6 float32 + 1 int32 = 28 bytes.
type parallax struct {
	TriAngleDeg, LateralAngleDeg, BaselineOverDepth, Baseline, MedianDepth, Focal float32
	NPoints                                                                       int32
}

// Lib is the dlopen'd shared library with the C API bound by name.
type Lib struct {
	handle uintptr

	optNew     func() uintptr
	optFree    func(uintptr)
	optSetDev  func(uintptr, string)
	load       func(string, uintptr) uintptr
	free       func(uintptr)
	lastErr    func(uintptr) uintptr
	geoOf      func(uintptr, unsafe.Pointer) int32
	run        func(uintptr, unsafe.Pointer, int32, int32, int32, unsafe.Pointer, unsafe.Pointer) int32
	bufFree    func(uintptr)
	abiVersion func() int32

	// accumulator + parallax (for in-process video->scene baking)
	accNew      func(int32, int32, float32) uintptr
	accFree     func(uintptr)
	accAddPair  func(uintptr, unsafe.Pointer, int32) int32
	accFrameCnt func(uintptr) int32
	accCloud    func(uintptr, unsafe.Pointer, unsafe.Pointer) int32
	accFuse     func(uintptr, float32, int32, int32, unsafe.Pointer, unsafe.Pointer) int32
	parallaxFn  func(unsafe.Pointer, int32, int32, int32, int32, float32, unsafe.Pointer) int32
}

// Engine is one loaded model (context) belonging to a Lib.
type Engine struct {
	mu  sync.Mutex
	lib *Lib
	ctx uintptr
	Geo geometry
}

// cstr reads a NUL-terminated C string at p (0 -> "").
func cstr(p uintptr) string {
	if p == 0 {
		return ""
	}
	var b []byte
	for i := uintptr(0); ; i++ {
		c := *(*byte)(unsafe.Pointer(p + i))
		if c == 0 {
			break
		}
		b = append(b, c)
	}
	return string(b)
}

// OpenLib dlopens libPath and binds the C API.
func OpenLib(libPath string) (*Lib, error) {
	handle, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("dlopen %s: %w", libPath, err)
	}
	l := &Lib{handle: handle}
	purego.RegisterLibFunc(&l.optNew, handle, "free_splatter_options_new")
	purego.RegisterLibFunc(&l.optFree, handle, "free_splatter_options_free")
	purego.RegisterLibFunc(&l.optSetDev, handle, "free_splatter_options_set_device")
	purego.RegisterLibFunc(&l.load, handle, "free_splatter_load")
	purego.RegisterLibFunc(&l.free, handle, "free_splatter_free")
	purego.RegisterLibFunc(&l.lastErr, handle, "free_splatter_last_error")
	purego.RegisterLibFunc(&l.geoOf, handle, "free_splatter_geometry_of")
	purego.RegisterLibFunc(&l.run, handle, "free_splatter_run")
	purego.RegisterLibFunc(&l.bufFree, handle, "free_splatter_buf_free")
	purego.RegisterLibFunc(&l.abiVersion, handle, "free_splatter_abi_version")
	purego.RegisterLibFunc(&l.accNew, handle, "free_splatter_accumulator_new")
	purego.RegisterLibFunc(&l.accFree, handle, "free_splatter_accumulator_free")
	purego.RegisterLibFunc(&l.accAddPair, handle, "free_splatter_accumulator_add_pair")
	purego.RegisterLibFunc(&l.accFrameCnt, handle, "free_splatter_accumulator_frame_count")
	purego.RegisterLibFunc(&l.accCloud, handle, "free_splatter_accumulator_cloud")
	purego.RegisterLibFunc(&l.accFuse, handle, "free_splatter_accumulator_fuse")
	purego.RegisterLibFunc(&l.parallaxFn, handle, "free_splatter_pair_parallax")
	if v := l.abiVersion(); v != 1 {
		purego.Dlclose(handle)
		return nil, fmt.Errorf("ABI mismatch: library reports %d, expected 1", v)
	}
	return l, nil
}

func (l *Lib) Close() {
	if l.handle != 0 {
		purego.Dlclose(l.handle)
		l.handle = 0
	}
}

// Load loads modelPath onto device ("" / "cpu" / "vulkan" / "vulkan:N" ...).
func (l *Lib) Load(modelPath, device string) (*Engine, error) {
	opts := l.optNew()
	if opts == 0 {
		return nil, fmt.Errorf("options_new failed (out of memory)")
	}
	if device != "" {
		l.optSetDev(opts, device)
	}
	ctx := l.load(modelPath, opts)
	l.optFree(opts)
	if ctx == 0 {
		return nil, fmt.Errorf("load %s: out of memory", modelPath)
	}
	if msg := cstr(l.lastErr(ctx)); msg != "" {
		l.free(ctx)
		return nil, fmt.Errorf("load %s: %s", modelPath, msg)
	}
	e := &Engine{lib: l, ctx: ctx}
	if l.geoOf(ctx, unsafe.Pointer(&e.Geo)) != 0 {
		l.free(ctx)
		return nil, fmt.Errorf("geometry_of failed")
	}
	return e, nil
}

// Run executes inference over nViews images laid out NCHW in [0,1] (length
// nViews*inChannels*H*W) and returns a copy of the activated gaussians
// (nViews*H*W*gaussianChannels float32). Serialized; the C buffer is freed here.
func (e *Engine) Run(images []float32, nViews int) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	var outPtr uintptr
	var nOut uint64
	ret := e.lib.run(e.ctx, unsafe.Pointer(&images[0]), int32(nViews),
		e.Geo.imageHeight, e.Geo.imageWidth,
		unsafe.Pointer(&outPtr), unsafe.Pointer(&nOut))
	runtime.KeepAlive(images)
	if ret != 0 {
		return nil, fmt.Errorf("run failed: %s", cstr(e.lib.lastErr(e.ctx)))
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(outPtr)), int(nOut))
	out := make([]float32, int(nOut))
	copy(out, src)
	e.lib.bufFree(outPtr)
	return out, nil
}

// Close frees the model context (not the shared library). NULL-safe / idempotent.
func (e *Engine) Close() {
	if e.ctx != 0 {
		e.lib.free(e.ctx)
		e.ctx = 0
	}
}

// --- accumulator + parallax helpers (in-process video->scene baking) ---

// pointsFromC copies a malloc'd free_splatter_point[n] to a Go slice and frees it.
func (l *Lib) pointsFromC(ptr uintptr, n uint64) []point {
	if ptr == 0 {
		return nil
	}
	out := make([]point, int(n))
	if n > 0 {
		copy(out, unsafe.Slice((*point)(unsafe.Pointer(ptr)), int(n)))
	}
	l.bufFree(ptr)
	return out
}

// accCloudCopy returns a copy of the accumulator's current cloud.
func (l *Lib) accCloudCopy(acc uintptr) []point {
	var ptr uintptr
	var n uint64
	if l.accCloud(acc, unsafe.Pointer(&ptr), unsafe.Pointer(&n)) != 0 {
		return nil
	}
	return l.pointsFromC(ptr, n)
}

// accFuseCopy returns the consensus-fused cloud (voxelFrac, k, mode: 0 avg/1 kept/2 best).
func (l *Lib) accFuseCopy(acc uintptr, voxelFrac float32, k, mode int32) []point {
	var ptr uintptr
	var n uint64
	if l.accFuse(acc, voxelFrac, k, mode, unsafe.Pointer(&ptr), unsafe.Pointer(&n)) != 0 {
		return nil
	}
	return l.pointsFromC(ptr, n)
}

// pairParallax computes the after-inference parallax of a 2-view gaussian buffer
// (the layout Engine.Run returns for nViews=2).
func (l *Lib) pairParallax(g []float32, h, w, gc int32, thr float32) (parallax, bool) {
	var px parallax
	rc := l.parallaxFn(unsafe.Pointer(&g[0]), 2, h, w, gc, thr, unsafe.Pointer(&px))
	runtime.KeepAlive(g)
	return px, rc == 0
}

// accAddPairSlice folds one pair's gaussians into the accumulator (the C side
// copies them, so g may be reused after). Returns 0 on success.
func (l *Lib) accAddPairSlice(acc uintptr, g []float32, gc int32) int32 {
	rc := l.accAddPair(acc, unsafe.Pointer(&g[0]), gc)
	runtime.KeepAlive(g)
	return rc
}
