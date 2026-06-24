// stb single-header implementations (decode + resize), isolated in their own TU
// so their warnings don't reach the rest of the tree. Built with -w (see
// CMakeLists.txt). Used by free_splatter-cli for image input.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
