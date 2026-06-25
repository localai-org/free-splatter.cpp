// convert.go — the numeric glue between uploaded photos and the WebGL viewer,
// kept in memory-safe Go so untrusted image bytes never touch the C/stb path.
//
//   preprocessImage: decoded photo -> model input (center-crop, resize, CHW)
//   encodeSplat:     activated gaussians -> antimatter15 .splat bytes
//
// Both mirror tools/free_splatter-cli.cpp; resampling need only be close (the
// network is robust), exact stb-parity is the test harness's job, not the demo's.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"sort"
)

const (
	gaussianChannels = 23
	shC0             = 0.28209479177387814
)

// preprocessImage decodes data, center-crops to a square, bilinear-resizes to
// size x size, scales to [0,1], and lays it out CHW (channel-major) — the layout
// free_splatter_run expects per view.
func preprocessImage(data []byte, size int) ([]float32, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return preprocessDecoded(img, size)
}

// preprocessDecoded lays out an already-decoded image (e.g. a background-removed
// one composited on white) as the model input: center-crop, bilinear resize to
// size, scale to [0,1], CHW.
func preprocessDecoded(img image.Image, size int) ([]float32, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w == 0 || h == 0 {
		return nil, fmt.Errorf("empty image")
	}
	// Flatten to RGBA bytes so we can sample by index regardless of source model.
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)

	s := w
	if h < s {
		s = h
	}
	left, top := (w-s)/2, (h-s)/2

	at := func(xx, yy, c int) float64 {
		if xx < 0 {
			xx = 0
		} else if xx > s-1 {
			xx = s - 1
		}
		if yy < 0 {
			yy = 0
		} else if yy > s-1 {
			yy = s - 1
		}
		return float64(rgba.Pix[(top+yy)*rgba.Stride+(left+xx)*4+c])
	}

	out := make([]float32, 3*size*size)
	plane := size * size
	for oy := 0; oy < size; oy++ {
		sy := (float64(oy)+0.5)*float64(s)/float64(size) - 0.5
		y0 := int(math.Floor(sy))
		fy := sy - float64(y0)
		for ox := 0; ox < size; ox++ {
			sx := (float64(ox)+0.5)*float64(s)/float64(size) - 0.5
			x0 := int(math.Floor(sx))
			fx := sx - float64(x0)
			for c := 0; c < 3; c++ {
				v00, v10 := at(x0, y0, c), at(x0+1, y0, c)
				v01, v11 := at(x0, y0+1, c), at(x0+1, y0+1, c)
				top := v00*(1-fx) + v10*fx
				bot := v01*(1-fx) + v11*fx
				out[c*plane+oy*size+ox] = float32((top*(1-fy) + bot*fy) / 255.0)
			}
		}
	}
	return out, nil
}

// encodeSplat converts activated gaussians [n*gaussianChannels] to antimatter15
// .splat bytes (32/splat: pos[3]f32, scale[3]f32, rgba[4]u8, rot[4]u8 w,x,y,z).
// Prunes opacity <= thr, sorts by importance (opacity*volume) so a reveal/
// progressive loader shows dominant structure first, caps to maxSplats (0=all),
// and bakes in the OpenCV (y down, z fwd) -> OpenGL (y up) flip: diag(1,-1,-1)
// on position, quaternion (w,x,y,z)->(-x,w,-z,y).
func encodeSplat(g []float32, opacThr float32, maxSplats int) ([]byte, int) {
	n := len(g) / gaussianChannels
	type kv struct {
		imp float32
		idx int
	}
	keep := make([]kv, 0, n)
	for i := 0; i < n; i++ {
		base := i * gaussianChannels
		op := g[base+15]
		if op <= opacThr {
			continue
		}
		vol := maxf(g[base+16], 1e-9) * maxf(g[base+17], 1e-9) * maxf(g[base+18], 1e-9)
		keep = append(keep, kv{op * vol, i})
	}
	sort.Slice(keep, func(a, b int) bool { return keep[a].imp > keep[b].imp })
	m := len(keep)
	if maxSplats > 0 && m > maxSplats {
		m = maxSplats
	}

	buf := make([]byte, m*32)
	for k := 0; k < m; k++ {
		x := g[keep[k].idx*gaussianChannels:]
		o := k * 32
		putf(buf[o:], x[0])
		putf(buf[o+4:], -x[1])
		putf(buf[o+8:], -x[2])
		putf(buf[o+12:], x[16])
		putf(buf[o+16:], x[17])
		putf(buf[o+20:], x[18])
		for c := 0; c < 3; c++ {
			buf[o+24+c] = u8(clamp01(0.5+shC0*float64(x[3+c])) * 255.0)
		}
		buf[o+27] = u8(clamp01(float64(x[15])) * 255.0)
		q := [4]float64{-float64(x[20]), float64(x[19]), -float64(x[22]), float64(x[21])}
		nrm := math.Sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]) + 1e-12
		for c := 0; c < 4; c++ {
			buf[o+28+c] = u8(q[c]/nrm*128.0 + 128.0)
		}
	}
	return buf, m
}

func putf(b []byte, v float32) { binary.LittleEndian.PutUint32(b, math.Float32bits(v)) }

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func u8(v float64) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}
