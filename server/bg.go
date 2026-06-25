// bg.go — optional background removal for the object reconstruction path.
//
// The object checkpoint expects an isolated object on a white background (its
// Objaverse training distribution). The official FreeSplatter pipeline segments
// the foreground first; we mirror that with a pluggable external matting command
// (e.g. rembg) configured via -bgremove-cmd. We keep it as a subprocess — not an
// FFI/cgo dependency — so the Go server stays a clean static binary and the
// matting toolchain (onnxruntime etc.) lives entirely outside it.
package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// removeBackgrounds runs the configured matting command over the uploaded images
// and returns each foreground composited on white, ready for preprocessDecoded.
// cmdTemplate is split on spaces; the tokens {in} and {out} are replaced with
// temp directories holding NNN.png inputs/outputs (rembg "p" batch convention),
// so the tool loads its model once for all views.
func removeBackgrounds(cmdTemplate string, uploads [][]byte) ([]image.Image, error) {
	tmp, err := os.MkdirTemp("", "fsbg-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	indir, outdir := filepath.Join(tmp, "in"), filepath.Join(tmp, "out")
	if err := os.MkdirAll(indir, 0o755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		return nil, err
	}

	// Normalize every upload to PNG so the matting tool sees a uniform format.
	for i, b := range uploads {
		img, _, err := image.Decode(bytes.NewReader(b))
		if err != nil {
			return nil, fmt.Errorf("decode view %d: %w", i, err)
		}
		f, err := os.Create(filepath.Join(indir, fmt.Sprintf("%03d.png", i)))
		if err != nil {
			return nil, err
		}
		err = png.Encode(f, img)
		f.Close()
		if err != nil {
			return nil, err
		}
	}

	args := strings.Fields(cmdTemplate)
	if len(args) == 0 {
		return nil, fmt.Errorf("empty bgremove command")
	}
	for i, a := range args {
		a = strings.ReplaceAll(a, "{in}", indir)
		a = strings.ReplaceAll(a, "{out}", outdir)
		args[i] = a
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("bgremove command failed: %v: %s", err, strings.TrimSpace(string(out)))
	}

	res := make([]image.Image, len(uploads))
	for i := range uploads {
		b, err := os.ReadFile(filepath.Join(outdir, fmt.Sprintf("%03d.png", i)))
		if err != nil {
			return nil, fmt.Errorf("read cutout %d (matting produced no output?): %w", i, err)
		}
		cut, err := png.Decode(bytes.NewReader(b))
		if err != nil {
			return nil, fmt.Errorf("decode cutout %d: %w", i, err)
		}
		res[i] = compositeOnWhite(cut)
	}
	return res, nil
}

// compositeOnWhite alpha-composites an image (typically RGBA with a transparent
// background from the matter) over an opaque white background. image/color's
// RGBA() returns alpha-premultiplied 16-bit values, so "over white" is just
// fg_premult + (1-alpha)*white.
func compositeOnWhite(img image.Image) image.Image {
	b := img.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, bl, a := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			inv := uint32(0xffff) - a
			out.SetRGBA(x, y, color.RGBA{
				R: uint8((r + inv) >> 8),
				G: uint8((g + inv) >> 8),
				B: uint8((bl + inv) >> 8),
				A: 255,
			})
		}
	}
	return out
}
