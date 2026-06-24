// Minimal purego (no cgo) smoke binding for libfree_splatter. Proves the flat C
// API loads and calls cleanly from Go via dlopen/dlsym.
//
//   cmake -B build/shared -DFREE_SPLATTER_BUILD_SHARED=ON && cmake --build build/shared
//   FREE_SPLATTER_LIB=build/shared/libfree_splatter.so \
//     go test ./bindings/go/...   (needs github.com/ebitengine/purego)
package freesplatter

import (
	"os"
	"testing"

	"github.com/ebitengine/purego"
)

func TestABISmoke(t *testing.T) {
	lib := os.Getenv("FREE_SPLATTER_LIB")
	if lib == "" {
		t.Skip("set FREE_SPLATTER_LIB to libfree_splatter.so")
	}
	h, err := purego.Dlopen(lib, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		t.Fatalf("dlopen: %v", err)
	}

	var abiVersion func() int32
	purego.RegisterLibFunc(&abiVersion, h, "free_splatter_abi_version")
	if v := abiVersion(); v != 1 {
		t.Fatalf("abi version = %d, want 1", v)
	}

	// Options builder round-trip (no model needed).
	var optionsNew func() uintptr
	var optionsSetDevice func(uintptr, string)
	var optionsFree func(uintptr)
	purego.RegisterLibFunc(&optionsNew, h, "free_splatter_options_new")
	purego.RegisterLibFunc(&optionsSetDevice, h, "free_splatter_options_set_device")
	purego.RegisterLibFunc(&optionsFree, h, "free_splatter_options_free")
	o := optionsNew()
	optionsSetDevice(o, "cpu")
	optionsFree(o)
	optionsFree(0) // NULL-safe
}
