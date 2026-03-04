// Headers

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <cuda.h>
#include <nvrtc.h>


// Initialisation

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path);
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);

// Arrays

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                      struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
CUdeviceptr futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                      struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);
struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
CUdeviceptr futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                      struct futhark_i32_1d *arr);
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr);

// Opaque values


// Entry points

int futhark_entry_main(struct futhark_context *ctx, int32_t *out0,
                       int32_t *out1, int32_t *out2,
                       struct futhark_f32_2d **out3,
                       struct futhark_i32_1d **out4,
                       struct futhark_i32_1d **out5,
                       struct futhark_f32_1d **out6,
                       struct futhark_i32_1d **out7, const int32_t in0, const
                       struct futhark_f32_2d *in1);

// Miscellaneous

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void futhark_panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    next_token(buf, bufsize);

    if (sscanf(buf, "%"SCNu64, &shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(int elem_size, unsigned char *elem) {
  for (int j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    int tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = fread(&bin_shape, sizeof(bin_shape), 1, stdin);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, stdin);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes(elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int end_of_input() {
  skipspaces();
  char token[2];
  next_token(token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 0; i < rank; i++) {
        printf("[%"PRIi64"]", shape[i]);
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    int64_t elem_size = expected_type->size;
    int num_elems_read = fread(dest, (size_t)elem_size, 1, stdin);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

#define __private
static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"default-group-size",
                                            required_argument, NULL, 7},
                                           {"default-num-groups",
                                            required_argument, NULL, 8},
                                           {"default-tile-size",
                                            required_argument, NULL, 9},
                                           {"default-threshold",
                                            required_argument, NULL, 10},
                                           {"print-sizes", no_argument, NULL,
                                            11}, {"size", required_argument,
                                                  NULL, 12}, {"tuning",
                                                              required_argument,
                                                              NULL, 13},
                                           {"dump-cuda", required_argument,
                                            NULL, 14}, {"load-cuda",
                                                        required_argument, NULL,
                                                        15}, {"dump-ptx",
                                                              required_argument,
                                                              NULL, 16},
                                           {"load-ptx", required_argument, NULL,
                                            17}, {"nvrtc-option",
                                                  required_argument, NULL, 18},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:b", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                futhark_panic(1, "Cannot open %s: %s\n", optarg,
                              strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                futhark_panic(1, "Need a positive number of runs, not %s\n",
                              optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 8)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 9)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 11) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_size_name(i),
                       futhark_get_size_class(i));
            exit(0);
        }
        if (ch == 12) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown size: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for size option: %s\n",
                              optarg);
        }
        if (ch == 13) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const
                                                               char *,
                                                               size_t)) futhark_context_config_set_size);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning from '%s': %s\n", optarg,
                              ret);
        }
        if (ch == 14) {
            futhark_context_config_dump_program_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 15)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 16) {
            futhark_context_config_dump_ptx_to(cfg, optarg);
            entry_point = NULL;
        }
        if (ch == 17)
            futhark_context_config_load_ptx_from(cfg, optarg);
        if (ch == 18)
            futhark_context_config_add_nvrtc_option(cfg, optarg);
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind -
                                                                       1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output] [--default-group-size INT] [--default-num-groups INT] [--default-tile-size INT] [--default-threshold INT] [--print-sizes] [--size NAME=INT] [--tuning FILE] [--dump-cuda FILE] [--load-cuda FILE] [--dump-ptx FILE] [--load-ptx FILE] [--nvrtc-option OPT]");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs = 0, profile_run = 0;
    
    // We do not want to profile all the initialisation.
    futhark_context_pause_profiling(ctx);
    // Declare and read input.
    set_binary_mode(stdin);
    
    int32_t read_value_2649;
    
    if (read_scalar(&i32_info, &read_value_2649) != 0)
        futhark_panic(1,
                      "Error when reading input #%d of type %s (errno: %s).\n",
                      0, i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_2650;
    int64_t read_shape_2651[2];
    float *read_arr_2652 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_2652, read_shape_2651, 2) !=
        0)
        futhark_panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
                      "[][]", f32_info.type_name, strerror(errno));
    if (end_of_input() != 0)
        futhark_panic(1, "Expected EOF on stdin after reading input for %s.\n",
                      "\"main\"");
    
    int32_t result_2653;
    int32_t result_2654;
    int32_t result_2655;
    struct futhark_f32_2d *result_2656;
    struct futhark_i32_1d *result_2657;
    struct futhark_i32_1d *result_2658;
    struct futhark_f32_1d *result_2659;
    struct futhark_i32_1d *result_2660;
    
    if (perform_warmup) {
        int r;
        
        ;
        assert((read_value_2650 = futhark_new_f32_2d(ctx, read_arr_2652,
                                                     read_shape_2651[0],
                                                     read_shape_2651[1])) != 0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_2653, &result_2654, &result_2655,
                               &result_2656, &result_2657, &result_2658,
                               &result_2659, &result_2660, read_value_2649,
                               read_value_2650);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_2650) == 0);
        ;
        ;
        ;
        assert(futhark_free_f32_2d(ctx, result_2656) == 0);
        assert(futhark_free_i32_1d(ctx, result_2657) == 0);
        assert(futhark_free_i32_1d(ctx, result_2658) == 0);
        assert(futhark_free_f32_1d(ctx, result_2659) == 0);
        assert(futhark_free_i32_1d(ctx, result_2660) == 0);
    }
    time_runs = 1;
    // Proper run.
    for (int run = 0; run < num_runs; run++) {
        // Only profile last run.
        profile_run = run == num_runs - 1;
        
        int r;
        
        ;
        assert((read_value_2650 = futhark_new_f32_2d(ctx, read_arr_2652,
                                                     read_shape_2651[0],
                                                     read_shape_2651[1])) != 0);
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        // Only profile last run.
        if (profile_run)
            futhark_context_unpause_profiling(ctx);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_2653, &result_2654, &result_2655,
                               &result_2656, &result_2657, &result_2658,
                               &result_2659, &result_2660, read_value_2649,
                               read_value_2650);
        if (r != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        if (futhark_context_sync(ctx) != 0)
            futhark_panic(1, "%s", futhark_context_get_error(ctx));
        ;
        if (profile_run)
            futhark_context_pause_profiling(ctx);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_2650) == 0);
        if (run < num_runs - 1) {
            ;
            ;
            ;
            assert(futhark_free_f32_2d(ctx, result_2656) == 0);
            assert(futhark_free_i32_1d(ctx, result_2657) == 0);
            assert(futhark_free_i32_1d(ctx, result_2658) == 0);
            assert(futhark_free_f32_1d(ctx, result_2659) == 0);
            assert(futhark_free_i32_1d(ctx, result_2660) == 0);
        }
    }
    ;
    free(read_arr_2652);
    if (binary_output)
        set_binary_mode(stdout);
    write_scalar(stdout, binary_output, &i32_info, &result_2653);
    printf("\n");
    write_scalar(stdout, binary_output, &i32_info, &result_2654);
    printf("\n");
    write_scalar(stdout, binary_output, &i32_info, &result_2655);
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_2656)[0] *
                            futhark_shape_f32_2d(ctx, result_2656)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_2656, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_2656), 2);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_2657)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_2657, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_2657), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_2658)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_2658, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_2658), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_2659)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_2659, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_2659), 1);
        free(arr);
    }
    printf("\n");
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_2660)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_2660, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_2660), 1);
        free(arr);
    }
    printf("\n");
    ;
    ;
    ;
    assert(futhark_free_f32_2d(ctx, result_2656) == 0);
    assert(futhark_free_i32_1d(ctx, result_2657) == 0);
    assert(futhark_free_i32_1d(ctx, result_2658) == 0);
    assert(futhark_free_f32_1d(ctx, result_2659) == 0);
    assert(futhark_free_i32_1d(ctx, result_2660) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "%s", error);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <cuda.h>
#include <nvrtc.h>

// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

static inline uint8_t add8(uint8_t x, uint8_t y)
{
    return x + y;
}
static inline uint16_t add16(uint16_t x, uint16_t y)
{
    return x + y;
}
static inline uint32_t add32(uint32_t x, uint32_t y)
{
    return x + y;
}
static inline uint64_t add64(uint64_t x, uint64_t y)
{
    return x + y;
}
static inline uint8_t sub8(uint8_t x, uint8_t y)
{
    return x - y;
}
static inline uint16_t sub16(uint16_t x, uint16_t y)
{
    return x - y;
}
static inline uint32_t sub32(uint32_t x, uint32_t y)
{
    return x - y;
}
static inline uint64_t sub64(uint64_t x, uint64_t y)
{
    return x - y;
}
static inline uint8_t mul8(uint8_t x, uint8_t y)
{
    return x * y;
}
static inline uint16_t mul16(uint16_t x, uint16_t y)
{
    return x * y;
}
static inline uint32_t mul32(uint32_t x, uint32_t y)
{
    return x * y;
}
static inline uint64_t mul64(uint64_t x, uint64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline bool ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline bool ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline bool ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline bool ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline bool ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline bool ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline bool ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline bool ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline bool slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline bool slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline bool slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline bool slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline bool sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline bool sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline bool sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline bool sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
#if defined(__OPENCL_VERSION__)
static int32_t futrts_popc8(int8_t x)
{
    return popcount(x);
}
static int32_t futrts_popc16(int16_t x)
{
    return popcount(x);
}
static int32_t futrts_popc32(int32_t x)
{
    return popcount(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return popcount(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_popc8(int8_t x)
{
    return __popc(zext_i8_i32(x));
}
static int32_t futrts_popc16(int16_t x)
{
    return __popc(zext_i16_i32(x));
}
static int32_t futrts_popc32(int32_t x)
{
    return __popc(x);
}
static int32_t futrts_popc64(int64_t x)
{
    return __popcll(x);
}
#else
static int32_t futrts_popc8(int8_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc16(int16_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc32(int32_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
static int32_t futrts_popc64(int64_t x)
{
    int c = 0;
    
    for (; x; ++c)
        x &= x - 1;
    return c;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    return mul_hi(a, b);
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    return mul_hi(a, b);
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mul_hi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul_hi(a, b);
}
#elif defined(__CUDA_ARCH__)
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    return mulhi(a, b);
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    return mul64hi(a, b);
}
#else
static uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)
{
    uint16_t aa = a;
    uint16_t bb = b;
    
    return aa * bb >> 8;
}
static uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)
{
    uint32_t aa = a;
    uint32_t bb = b;
    
    return aa * bb >> 16;
}
static uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)
{
    uint64_t aa = a;
    uint64_t bb = b;
    
    return aa * bb >> 32;
}
static uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)
{
    __uint128_t aa = a;
    __uint128_t bb = b;
    
    return aa * bb >> 64;
}
#endif
#if defined(__OPENCL_VERSION__)
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return mad_hi(a, b, c);
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return mad_hi(a, b, c);
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return mad_hi(a, b, c);
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return mad_hi(a, b, c);
}
#else
static uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)
{
    return futrts_mul_hi8(a, b) + c;
}
static uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)
{
    return futrts_mul_hi16(a, b) + c;
}
static uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)
{
    return futrts_mul_hi32(a, b) + c;
}
static uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)
{
    return futrts_mul_hi64(a, b) + c;
}
#endif
#if defined(__OPENCL_VERSION__)
static int32_t futrts_clzz8(int8_t x)
{
    return clz(x);
}
static int32_t futrts_clzz16(int16_t x)
{
    return clz(x);
}
static int32_t futrts_clzz32(int32_t x)
{
    return clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return clz(x);
}
#elif defined(__CUDA_ARCH__)
static int32_t futrts_clzz8(int8_t x)
{
    return __clz(zext_i8_i32(x)) - 24;
}
static int32_t futrts_clzz16(int16_t x)
{
    return __clz(zext_i16_i32(x)) - 16;
}
static int32_t futrts_clzz32(int32_t x)
{
    return __clz(x);
}
static int32_t futrts_clzz64(int64_t x)
{
    return __clzll(x);
}
#else
static int32_t futrts_clzz8(int8_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz16(int16_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz32(int32_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
static int32_t futrts_clzz64(int64_t x)
{
    int n = 0;
    int bits = sizeof(x) * 8;
    
    for (int i = 0; i < bits; i++) {
        if (x < 0)
            break;
        n++;
        x <<= 1;
    }
    return n;
}
#endif
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return fmin(x, y);
}
static inline float fmax32(float x, float y)
{
    return fmax(x, y);
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline bool cmplt32(float x, float y)
{
    return x < y;
}
static inline bool cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return (float) x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return (float) x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return (float) x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return (float) x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return (float) x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return (float) x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return (float) x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return (float) x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return (uint64_t) x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return fmin(x, y);
}
static inline double fmax64(double x, double y)
{
    return fmax(x, y);
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline bool cmplt64(double x, double y)
{
    return x < y;
}
static inline bool cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return (double) x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return (double) x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return (double) x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return (double) x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return (double) x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return (double) x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return (double) x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return (double) x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return (int8_t) x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return (int16_t) x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return (int32_t) x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return (int64_t) x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return (uint8_t) x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return (uint16_t) x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return (uint32_t) x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return (uint64_t) x;
}
static inline float fpconv_f32_f32(float x)
{
    return (float) x;
}
static inline double fpconv_f32_f64(float x)
{
    return (double) x;
}
static inline float fpconv_f64_f32(double x)
{
    return (float) x;
}
static inline double fpconv_f64_f64(double x)
{
    return (double) x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_cosh32(float x)
{
    return cosh(x);
}
static inline float futrts_sinh32(float x)
{
    return sinh(x);
}
static inline float futrts_tanh32(float x)
{
    return tanh(x);
}
static inline float futrts_acosh32(float x)
{
    return acosh(x);
}
static inline float futrts_asinh32(float x)
{
    return asinh(x);
}
static inline float futrts_atanh32(float x)
{
    return atanh(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline bool futrts_isnan32(float x)
{
    return isnan(x);
}
static inline bool futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
#ifdef __OPENCL_VERSION__
static inline float fmod32(float x, float y)
{
    return fmod(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline float futrts_floor32(float x)
{
    return floor(x);
}
static inline float futrts_ceil32(float x)
{
    return ceil(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return mix(v0, v1, t);
}
static inline float futrts_mad32(float a, float b, float c)
{
    return mad(a, b, c);
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fma(a, b, c);
}
#else
static inline float fmod32(float x, float y)
{
    return fmodf(x, y);
}
static inline float futrts_round32(float x)
{
    return rintf(x);
}
static inline float futrts_floor32(float x)
{
    return floorf(x);
}
static inline float futrts_ceil32(float x)
{
    return ceilf(x);
}
static inline float futrts_lerp32(float v0, float v1, float t)
{
    return v0 + (v1 - v0) * t;
}
static inline float futrts_mad32(float a, float b, float c)
{
    return a * b + c;
}
static inline float futrts_fma32(float a, float b, float c)
{
    return fmaf(a, b, c);
}
#endif
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_cosh64(double x)
{
    return cosh(x);
}
static inline double futrts_sinh64(double x)
{
    return sinh(x);
}
static inline double futrts_tanh64(double x)
{
    return tanh(x);
}
static inline double futrts_acosh64(double x)
{
    return acosh(x);
}
static inline double futrts_asinh64(double x)
{
    return asinh(x);
}
static inline double futrts_atanh64(double x)
{
    return atanh(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_fma64(double a, double b, double c)
{
    return fma(a, b, c);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline double futrts_ceil64(double x)
{
    return ceil(x);
}
static inline double futrts_floor64(double x)
{
    return floor(x);
}
static inline bool futrts_isnan64(double x)
{
    return isnan(x);
}
static inline bool futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double fmod64(double x, double y)
{
    return fmod(x, y);
}
#ifdef __OPENCL_VERSION__
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return mix(v0, v1, t);
}
static inline double futrts_mad64(double a, double b, double c)
{
    return mad(a, b, c);
}
#else
static inline double futrts_lerp64(double v0, double v1, double t)
{
    return v0 + (v1 - v0) * t;
}
static inline double futrts_mad64(double a, double b, double c)
{
    return a * b + c;
}
#endif
int init_constants(struct futhark_context *);
int free_constants(struct futhark_context *);
static int32_t counter_mem_realtype_2435[10240];
static int32_t counter_mem_realtype_2454[10240];
struct memblock_device {
    int *references;
    CUdeviceptr mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
#include <cuda.h>
#include <nvrtc.h>
typedef CUdeviceptr fl_mem_t;
// Start of free_list.h.

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct free_list_entry {
  size_t size;
  fl_mem_t mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries;        // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
static void free_list_pack(struct free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem_t mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
static int free_list_find(struct free_list *l, const char *tag, size_t *size_out, fl_mem_t *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
static int free_list_first(struct free_list *l, fl_mem_t *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

// End of free_list.h.

// Start of cuda.h.

#define CUDA_SUCCEED(x) cuda_api_succeed(x, #x, __FILE__, __LINE__)
#define NVRTC_SUCCEED(x) nvrtc_api_succeed(x, #x, __FILE__, __LINE__)

static inline void cuda_api_succeed(CUresult res, const char *call,
    const char *file, int line) {
  if (res != CUDA_SUCCESS) {
    const char *err_str;
    cuGetErrorString(res, &err_str);
    if (err_str == NULL) { err_str = "Unknown"; }
    futhark_panic(-1, "%s:%d: CUDA call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

static inline void nvrtc_api_succeed(nvrtcResult res, const char *call,
                                     const char *file, int line) {
  if (res != NVRTC_SUCCESS) {
    const char *err_str = nvrtcGetErrorString(res);
    futhark_panic(-1, "%s:%d: NVRTC call\n  %s\nfailed with error code %d (%s)\n",
        file, line, call, res, err_str);
  }
}

struct cuda_config {
  int debugging;
  int logging;
  const char *preferred_device;

  const char *dump_program_to;
  const char *load_program_from;

  const char *dump_ptx_to;
  const char *load_ptx_from;

  size_t default_block_size;
  size_t default_grid_size;
  size_t default_tile_size;
  size_t default_threshold;

  int default_block_size_changed;
  int default_grid_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  const char **size_vars;
  size_t *size_values;
  const char **size_classes;
};

static void cuda_config_init(struct cuda_config *cfg,
                             int num_sizes,
                             const char *size_names[],
                             const char *size_vars[],
                             size_t *size_values,
                             const char *size_classes[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device = "";

  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;

  cfg->dump_ptx_to = NULL;
  cfg->load_ptx_from = NULL;

  cfg->default_block_size = 256;
  cfg->default_grid_size = 256;
  cfg->default_tile_size = 32;
  cfg->default_threshold = 32*1024;

  cfg->default_block_size_changed = 0;
  cfg->default_grid_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_vars = size_vars;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
}

struct cuda_context {
  CUdevice dev;
  CUcontext cu_ctx;
  CUmodule module;

  struct cuda_config cfg;

  struct free_list free_list;

  size_t max_block_size;
  size_t max_grid_size;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_shared_memory;
  size_t max_bespoke;

  size_t lockstep_width;
};

#define CU_DEV_ATTR(x) (CU_DEVICE_ATTRIBUTE_##x)
#define device_query(dev,attrib) _device_query(dev, CU_DEV_ATTR(attrib))
static int _device_query(CUdevice dev, CUdevice_attribute attrib) {
  int val;
  CUDA_SUCCEED(cuDeviceGetAttribute(&val, attrib, dev));
  return val;
}

#define CU_FUN_ATTR(x) (CU_FUNC_ATTRIBUTE_##x)
#define function_query(fn,attrib) _function_query(dev, CU_FUN_ATTR(attrib))
static int _function_query(CUfunction dev, CUfunction_attribute attrib) {
  int val;
  CUDA_SUCCEED(cuFuncGetAttribute(&val, attrib, dev));
  return val;
}

static void set_preferred_device(struct cuda_config *cfg, const char *s) {
  cfg->preferred_device = s;
}

static int cuda_device_setup(struct cuda_context *ctx) {
  char name[256];
  int count, chosen = -1, best_cc = -1;
  int cc_major_best, cc_minor_best;
  int cc_major, cc_minor;
  CUdevice dev;

  CUDA_SUCCEED(cuDeviceGetCount(&count));
  if (count == 0) { return 1; }

  // XXX: Current device selection policy is to choose the device with the
  // highest compute capability (if no preferred device is set).
  // This should maybe be changed, since greater compute capability is not
  // necessarily an indicator of better performance.
  for (int i = 0; i < count; i++) {
    CUDA_SUCCEED(cuDeviceGet(&dev, i));

    cc_major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
    cc_minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

    CUDA_SUCCEED(cuDeviceGetName(name, sizeof(name) - 1, dev));
    name[sizeof(name) - 1] = 0;

    if (ctx->cfg.debugging) {
      fprintf(stderr, "Device #%d: name=\"%s\", compute capability=%d.%d\n",
          i, name, cc_major, cc_minor);
    }

    if (device_query(dev, COMPUTE_MODE) == CU_COMPUTEMODE_PROHIBITED) {
      if (ctx->cfg.debugging) {
        fprintf(stderr, "Device #%d is compute-prohibited, ignoring\n", i);
      }
      continue;
    }

    if (best_cc == -1 || cc_major > cc_major_best ||
        (cc_major == cc_major_best && cc_minor > cc_minor_best)) {
      best_cc = i;
      cc_major_best = cc_major;
      cc_minor_best = cc_minor;
    }

    if (chosen == -1 && strstr(name, ctx->cfg.preferred_device) == name) {
      chosen = i;
    }
  }

  if (chosen == -1) { chosen = best_cc; }
  if (chosen == -1) { return 1; }

  if (ctx->cfg.debugging) {
    fprintf(stderr, "Using device #%d\n", chosen);
  }

  CUDA_SUCCEED(cuDeviceGet(&ctx->dev, chosen));
  return 0;
}

static char *concat_fragments(const char *src_fragments[]) {
  size_t src_len = 0;
  const char **p;

  for (p = src_fragments; *p; p++) {
    src_len += strlen(*p);
  }

  char *src = (char*) malloc(src_len + 1);
  size_t n = 0;
  for (p = src_fragments; *p; p++) {
    strcpy(src + n, *p);
    n += strlen(*p);
  }

  return src;
}

static const char *cuda_nvrtc_get_arch(CUdevice dev) {
  struct {
    int major;
    int minor;
    const char *arch_str;
  } static const x[] = {
    { 3, 0, "compute_30" },
    { 3, 2, "compute_32" },
    { 3, 5, "compute_35" },
    { 3, 7, "compute_37" },
    { 5, 0, "compute_50" },
    { 5, 2, "compute_52" },
    { 5, 3, "compute_53" },
    { 6, 0, "compute_60" },
    { 6, 1, "compute_61" },
    { 6, 2, "compute_62" },
    { 7, 0, "compute_70" },
    { 7, 2, "compute_72" },
    { 7, 5, "compute_75" }
  };

  int major = device_query(dev, COMPUTE_CAPABILITY_MAJOR);
  int minor = device_query(dev, COMPUTE_CAPABILITY_MINOR);

  int chosen = -1;
  for (int i = 0; i < sizeof(x)/sizeof(x[0]); i++) {
    if (x[i].major < major || (x[i].major == major && x[i].minor <= minor)) {
      chosen = i;
    } else {
      break;
    }
  }

  if (chosen == -1) {
    futhark_panic(-1, "Unsupported compute capability %d.%d\n", major, minor);
  }

  if (x[chosen].major != major || x[chosen].minor != minor) {
    fprintf(stderr,
            "Warning: device compute capability is %d.%d, but newest supported by Futhark is %d.%d.\n",
            major, minor, x[chosen].major, x[chosen].minor);
  }

  return x[chosen].arch_str;
}

static char *cuda_nvrtc_build(struct cuda_context *ctx, const char *src,
                              const char *extra_opts[]) {
  nvrtcProgram prog;
  NVRTC_SUCCEED(nvrtcCreateProgram(&prog, src, "futhark-cuda", 0, NULL, NULL));
  int arch_set = 0, num_extra_opts;

  // nvrtc cannot handle multiple -arch options.  Hence, if one of the
  // extra_opts is -arch, we have to be careful not to do our usual
  // automatic generation.
  for (num_extra_opts = 0; extra_opts[num_extra_opts] != NULL; num_extra_opts++) {
    if (strstr(extra_opts[num_extra_opts], "-arch")
        == extra_opts[num_extra_opts] ||
        strstr(extra_opts[num_extra_opts], "--gpu-architecture")
        == extra_opts[num_extra_opts]) {
      arch_set = 1;
    }
  }

  size_t n_opts, i = 0, i_dyn, n_opts_alloc = 20 + num_extra_opts + ctx->cfg.num_sizes;
  const char **opts = (const char**) malloc(n_opts_alloc * sizeof(const char *));
  if (!arch_set) {
    opts[i++] = "-arch";
    opts[i++] = cuda_nvrtc_get_arch(ctx->dev);
  }
  opts[i++] = "-default-device";
  if (ctx->cfg.debugging) {
    opts[i++] = "-G";
    opts[i++] = "-lineinfo";
  } else {
    opts[i++] = "--disable-warnings";
  }
  i_dyn = i;
  for (size_t j = 0; j < ctx->cfg.num_sizes; j++) {
    opts[i++] = msgprintf("-D%s=%zu", ctx->cfg.size_vars[j],
        ctx->cfg.size_values[j]);
  }
  opts[i++] = msgprintf("-DLOCKSTEP_WIDTH=%zu", ctx->lockstep_width);
  opts[i++] = msgprintf("-DMAX_THREADS_PER_BLOCK=%zu", ctx->max_block_size);

  // It is crucial that the extra_opts are last, so that the free()
  // logic below does not cause problems.
  for (int j = 0; extra_opts[j] != NULL; j++) {
    opts[i++] = extra_opts[j];
  }

  n_opts = i;

  if (ctx->cfg.debugging) {
    fprintf(stderr, "NVRTC compile options:\n");
    for (size_t j = 0; j < n_opts; j++) {
      fprintf(stderr, "\t%s\n", opts[j]);
    }
    fprintf(stderr, "\n");
  }

  nvrtcResult res = nvrtcCompileProgram(prog, n_opts, opts);
  if (res != NVRTC_SUCCESS) {
    size_t log_size;
    if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS) {
      char *log = (char*) malloc(log_size);
      if (nvrtcGetProgramLog(prog, log) == NVRTC_SUCCESS) {
        fprintf(stderr,"Compilation log:\n%s\n", log);
      }
      free(log);
    }
    NVRTC_SUCCEED(res);
  }

  for (i = i_dyn; i < n_opts-num_extra_opts; i++) { free((char *)opts[i]); }
  free(opts);

  char *ptx;
  size_t ptx_size;
  NVRTC_SUCCEED(nvrtcGetPTXSize(prog, &ptx_size));
  ptx = (char*) malloc(ptx_size);
  NVRTC_SUCCEED(nvrtcGetPTX(prog, ptx));

  NVRTC_SUCCEED(nvrtcDestroyProgram(&prog));

  return ptx;
}

static void cuda_size_setup(struct cuda_context *ctx)
{
  if (ctx->cfg.default_block_size > ctx->max_block_size) {
    if (ctx->cfg.default_block_size_changed) {
      fprintf(stderr,
          "Note: Device limits default block size to %zu (down from %zu).\n",
          ctx->max_block_size, ctx->cfg.default_block_size);
    }
    ctx->cfg.default_block_size = ctx->max_block_size;
  }
  if (ctx->cfg.default_grid_size > ctx->max_grid_size) {
    if (ctx->cfg.default_grid_size_changed) {
      fprintf(stderr,
          "Note: Device limits default grid size to %zu (down from %zu).\n",
          ctx->max_grid_size, ctx->cfg.default_grid_size);
    }
    ctx->cfg.default_grid_size = ctx->max_grid_size;
  }
  if (ctx->cfg.default_tile_size > ctx->max_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr,
          "Note: Device limits default tile size to %zu (down from %zu).\n",
          ctx->max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = ctx->max_tile_size;
  }

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class, *size_name;
    size_t *size_value, max_value, default_value;

    size_class = ctx->cfg.size_classes[i];
    size_value = &ctx->cfg.size_values[i];
    size_name = ctx->cfg.size_names[i];

    if (strstr(size_class, "group_size") == size_class) {
      max_value = ctx->max_block_size;
      default_value = ctx->cfg.default_block_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = ctx->max_grid_size;
      default_value = ctx->cfg.default_grid_size;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = ctx->max_tile_size;
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = ctx->max_threshold;
      default_value = ctx->cfg.default_threshold;
    } else {
      // Bespoke sizes have no limit or default.
      max_value = 0;
    }

    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %zu (down from %zu)\n",
              size_name, max_value, *size_value);
      *size_value = max_value;
    }
  }
}

static void dump_string_to_file(const char *file, const char *buf) {
  FILE *f = fopen(file, "w");
  assert(f != NULL);
  assert(fputs(buf, f) != EOF);
  assert(fclose(f) == 0);
}

static void load_string_from_file(const char *file, char **obuf, size_t *olen) {
  char *buf;
  size_t len;
  FILE *f = fopen(file, "r");

  assert(f != NULL);
  assert(fseek(f, 0, SEEK_END) == 0);
  len = ftell(f);
  assert(fseek(f, 0, SEEK_SET) == 0);

  buf = (char*) malloc(len + 1);
  assert(fread(buf, 1, len, f) == len);
  buf[len] = 0;
  *obuf = buf;
  if (olen != NULL) {
    *olen = len;
  }

  assert(fclose(f) == 0);
}

static void cuda_module_setup(struct cuda_context *ctx,
                              const char *src_fragments[],
                              const char *extra_opts[]) {
  char *ptx = NULL, *src = NULL;

  if (ctx->cfg.load_ptx_from == NULL && ctx->cfg.load_program_from == NULL) {
    src = concat_fragments(src_fragments);
    ptx = cuda_nvrtc_build(ctx, src, extra_opts);
  } else if (ctx->cfg.load_ptx_from == NULL) {
    load_string_from_file(ctx->cfg.load_program_from, &src, NULL);
    ptx = cuda_nvrtc_build(ctx, src, extra_opts);
  } else {
    if (ctx->cfg.load_program_from != NULL) {
      fprintf(stderr,
              "WARNING: Loading PTX from %s instead of C code from %s\n",
              ctx->cfg.load_ptx_from, ctx->cfg.load_program_from);
    }

    load_string_from_file(ctx->cfg.load_ptx_from, &ptx, NULL);
  }

  if (ctx->cfg.dump_program_to != NULL) {
    if (src == NULL) {
      src = concat_fragments(src_fragments);
    }
    dump_string_to_file(ctx->cfg.dump_program_to, src);
  }
  if (ctx->cfg.dump_ptx_to != NULL) {
    dump_string_to_file(ctx->cfg.dump_ptx_to, ptx);
  }

  CUDA_SUCCEED(cuModuleLoadData(&ctx->module, ptx));

  free(ptx);
  if (src != NULL) {
    free(src);
  }
}

static void cuda_setup(struct cuda_context *ctx, const char *src_fragments[], const char *extra_opts[]) {
  CUDA_SUCCEED(cuInit(0));

  if (cuda_device_setup(ctx) != 0) {
    futhark_panic(-1, "No suitable CUDA device found.\n");
  }
  CUDA_SUCCEED(cuCtxCreate(&ctx->cu_ctx, 0, ctx->dev));

  free_list_init(&ctx->free_list);

  ctx->max_shared_memory = device_query(ctx->dev, MAX_SHARED_MEMORY_PER_BLOCK);
  ctx->max_block_size = device_query(ctx->dev, MAX_THREADS_PER_BLOCK);
  ctx->max_grid_size = device_query(ctx->dev, MAX_GRID_DIM_X);
  ctx->max_tile_size = sqrt(ctx->max_block_size);
  ctx->max_threshold = 0;
  ctx->max_bespoke = 0;
  ctx->lockstep_width = device_query(ctx->dev, WARP_SIZE);

  cuda_size_setup(ctx);
  cuda_module_setup(ctx, src_fragments, extra_opts);
}

static CUresult cuda_free_all(struct cuda_context *ctx);

static void cuda_cleanup(struct cuda_context *ctx) {
  CUDA_SUCCEED(cuda_free_all(ctx));
  CUDA_SUCCEED(cuModuleUnload(ctx->module));
  CUDA_SUCCEED(cuCtxDestroy(ctx->cu_ctx));
}

static CUresult cuda_alloc(struct cuda_context *ctx, size_t min_size,
                           const char *tag, CUdeviceptr *mem_out) {
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;
  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    if (size >= min_size) {
      return CUDA_SUCCESS;
    } else {
      CUresult res = cuMemFree(*mem_out);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    }
  }

  CUresult res = cuMemAlloc(mem_out, min_size);
  while (res == CUDA_ERROR_OUT_OF_MEMORY) {
    CUdeviceptr mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      res = cuMemFree(mem);
      if (res != CUDA_SUCCESS) {
        return res;
      }
    } else {
      break;
    }
    res = cuMemAlloc(mem_out, min_size);
  }

  return res;
}

static CUresult cuda_free(struct cuda_context *ctx, CUdeviceptr mem,
                          const char *tag) {
  size_t size;
  CUdeviceptr existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    CUresult res = cuMemFree(existing_mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  CUresult res = cuMemGetAddressRange(NULL, &size, mem);
  if (res == CUDA_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return res;
}

static CUresult cuda_free_all(struct cuda_context *ctx) {
  CUdeviceptr mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    CUresult res = cuMemFree(mem);
    if (res != CUDA_SUCCESS) {
      return res;
    }
  }

  return CUDA_SUCCESS;
}

// End of cuda.h.

const char *cuda_program[] =
           {"#define FUTHARK_CUDA\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long long int64_t;\ntypedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\ntypedef uint8_t uchar;\ntypedef uint16_t ushort;\ntypedef uint32_t uint;\ntypedef uint64_t ulong;\n#define __kernel extern \"C\" __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK)\n#define __global\n#define __local\n#define __private\n#define __constant\n#define __write_only\n#define __read_only\nstatic inline int get_group_id_fn(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return blockIdx.x;\n        \n      case 1:\n        return blockIdx.y;\n        \n      case 2:\n        return blockIdx.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_group_id(d) get_group_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_num_groups_fn(int block_dim0, int block_dim1,\n                                    int block_dim2, int d)\n{\n    switch (d) {\n        \n      case 0:\n        d = block_dim0;\n        break;\n        \n      case 1:\n        d = block_dim1;\n        break;\n        \n      case 2:\n        d = block_dim2;\n        break;\n    }\n    switch (d) {\n        \n      case 0:\n        return gridDim.x;\n        \n      case 1:\n        return gridDim.y;\n        \n      case 2:\n        return gridDim.z;\n        \n      default:\n        return 0;\n    }\n}\n#define get_num_groups(d) get_num_groups_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_local_id(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return threadIdx.x;\n        \n      case 1:\n        return threadIdx.y;\n        \n      case 2:\n        return threadI",
            "dx.z;\n        \n      default:\n        return 0;\n    }\n}\nstatic inline int get_local_size(int d)\n{\n    switch (d) {\n        \n      case 0:\n        return blockDim.x;\n        \n      case 1:\n        return blockDim.y;\n        \n      case 2:\n        return blockDim.z;\n        \n      default:\n        return 0;\n    }\n}\nstatic inline int get_global_id_fn(int block_dim0, int block_dim1,\n                                   int block_dim2, int d)\n{\n    return get_group_id(d) * get_local_size(d) + get_local_id(d);\n}\n#define get_global_id(d) get_global_id_fn(block_dim0, block_dim1, block_dim2, d)\nstatic inline int get_global_size(int block_dim0, int block_dim1,\n                                  int block_dim2, int d)\n{\n    return get_num_groups(d) * get_local_size(d);\n}\n#define CLK_LOCAL_MEM_FENCE 1\n#define CLK_GLOBAL_MEM_FENCE 2\nstatic inline void barrier(int x)\n{\n    __syncthreads();\n}\nstatic inline void mem_fence_local()\n{\n    __threadfence_block();\n}\nstatic inline void mem_fence_global()\n{\n    __threadfence();\n}\n#define NAN (0.0/0.0)\n#define INFINITY (1.0/0.0)\nextern volatile __shared__ char shared_mem[];\nstatic inline uint8_t add8(uint8_t x, uint8_t y)\n{\n    return x + y;\n}\nstatic inline uint16_t add16(uint16_t x, uint16_t y)\n{\n    return x + y;\n}\nstatic inline uint32_t add32(uint32_t x, uint32_t y)\n{\n    return x + y;\n}\nstatic inline uint64_t add64(uint64_t x, uint64_t y)\n{\n    return x + y;\n}\nstatic inline uint8_t sub8(uint8_t x, uint8_t y)\n{\n    return x - y;\n}\nstatic inline uint16_t sub16(uint16_t x, uint16_t y)\n{\n    return x - y;\n}\nstatic inline uint32_t sub32(uint32_t x, uint32_t y)\n{\n    return x - y;\n}\nstatic inline uint64_t sub64(uint64_t x, uint64_t y)\n{\n    return x - y;\n}\nstatic inline uint8_t mul8(uint8_t x, uint8_t y)\n{\n    return x * y;\n}\nstatic inline uint16_t mul16(uint16_t x, uint16_t y)\n{\n    return x * y;\n}\nstatic inline uint32_t mul32(uint32_t x, uint32_t y)\n{\n    return x * y;\n}\nstatic inline uint64_t mul64(uint64_t x, uint64_t y)\n{\n    return x * y;\n",
            "}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstati",
            "c inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, ",
            "uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline bool ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline bool ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline bool ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline bool ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline bool ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline bool ule32(uint32_t x, uint32_t y)\n{\n    return x",
            " <= y;\n}\nstatic inline bool ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline bool slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline bool slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline bool slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline bool slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline bool sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline bool sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline bool itob_i8_bool(int8_t x)\n{\n    return x;\n}\nstatic inline bool itob_i16_bool(int16_t x)\n{\n    return x;\n}\nstatic inline bool itob_i32_bool(int32_t x)\n{\n    return x;\n}\nstatic inline bool itob_i64_bool(int64_t x)\n{\n    return x;\n}\nstatic inline int8_t btoi_bool_i8(bool x)\n{\n    return x;\n}\nstatic inline int16_t btoi_bool_i16(bool x)\n{\n    return x;\n}\nstatic inline int32_t btoi_bool_i32(bool x)\n{\n    return x;\n}\nstatic inline int64_t btoi_bool_i64(bool x)\n{\n    return x;\n}\n#define sext_i8_i8(x) ((int8_t) (i",
            "nt8_t) x)\n#define sext_i8_i16(x) ((int16_t) (int8_t) x)\n#define sext_i8_i32(x) ((int32_t) (int8_t) x)\n#define sext_i8_i64(x) ((int64_t) (int8_t) x)\n#define sext_i16_i8(x) ((int8_t) (int16_t) x)\n#define sext_i16_i16(x) ((int16_t) (int16_t) x)\n#define sext_i16_i32(x) ((int32_t) (int16_t) x)\n#define sext_i16_i64(x) ((int64_t) (int16_t) x)\n#define sext_i32_i8(x) ((int8_t) (int32_t) x)\n#define sext_i32_i16(x) ((int16_t) (int32_t) x)\n#define sext_i32_i32(x) ((int32_t) (int32_t) x)\n#define sext_i32_i64(x) ((int64_t) (int32_t) x)\n#define sext_i64_i8(x) ((int8_t) (int64_t) x)\n#define sext_i64_i16(x) ((int16_t) (int64_t) x)\n#define sext_i64_i32(x) ((int32_t) (int64_t) x)\n#define sext_i64_i64(x) ((int64_t) (int64_t) x)\n#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)\n#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)\n#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)\n#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)\n#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)\n#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)\n#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)\n#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)\n#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)\n#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)\n#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)\n#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)\n#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)\n#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)\n#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)\n#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    return popcount(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return popcount(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_popc8(int8_t x)\n{\n    return __popc(zext_i8_i32(x));\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    return __popc(zext_i16_i32(x));\n}\nstatic int",
            "32_t futrts_popc32(int32_t x)\n{\n    return __popc(x);\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    return __popcll(x);\n}\n#else\nstatic int32_t futrts_popc8(int8_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc16(int16_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc32(int32_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\nstatic int32_t futrts_popc64(int64_t x)\n{\n    int c = 0;\n    \n    for (; x; ++c)\n        x &= x - 1;\n    return c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mul_hi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul_hi(a, b);\n}\n#elif defined(__CUDA_ARCH__)\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    return mulhi(a, b);\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    return mul64hi(a, b);\n}\n#else\nstatic uint8_t futrts_mul_hi8(uint8_t a, uint8_t b)\n{\n    uint16_t aa = a;\n    uint16_t bb = b;\n    \n    return aa * bb >> 8;\n}\nstatic uint16_t futrts_mul_hi16(uint16_t a, uint16_t b)\n{\n    uint32_t aa = a;\n    uint32_t bb = b;\n    \n    return aa * bb >> 16;\n}\nstatic uint32_t futrts_mul_hi32(uint32_t a, uint32_t b)\n{\n    uint64_t aa = a;\n    uint64_t bb = b;\n    \n    return aa * bb >> 32;\n}\nstatic uint64_t futrts_mul_hi64(uint64_t a, uint64_t b)\n{\n    __uint128_t aa = a;\n    __uint128_t bb = b;\n    \n    return aa * bb >> 64;\n}\n#endif\n#if defined(__OPENCL_VE",
            "RSION__)\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return mad_hi(a, b, c);\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return mad_hi(a, b, c);\n}\n#else\nstatic uint8_t futrts_mad_hi8(uint8_t a, uint8_t b, uint8_t c)\n{\n    return futrts_mul_hi8(a, b) + c;\n}\nstatic uint16_t futrts_mad_hi16(uint16_t a, uint16_t b, uint16_t c)\n{\n    return futrts_mul_hi16(a, b) + c;\n}\nstatic uint32_t futrts_mad_hi32(uint32_t a, uint32_t b, uint32_t c)\n{\n    return futrts_mul_hi32(a, b) + c;\n}\nstatic uint64_t futrts_mad_hi64(uint64_t a, uint64_t b, uint64_t c)\n{\n    return futrts_mul_hi64(a, b) + c;\n}\n#endif\n#if defined(__OPENCL_VERSION__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return clz(x);\n}\n#elif defined(__CUDA_ARCH__)\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    return __clz(zext_i8_i32(x)) - 24;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    return __clz(zext_i16_i32(x)) - 16;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    return __clz(x);\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    return __clzll(x);\n}\n#else\nstatic int32_t futrts_clzz8(int8_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz16(int16_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz32(int32_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for",
            " (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\nstatic int32_t futrts_clzz64(int64_t x)\n{\n    int n = 0;\n    int bits = sizeof(x) * 8;\n    \n    for (int i = 0; i < bits; i++) {\n        if (x < 0)\n            break;\n        n++;\n        x <<= 1;\n    }\n    return n;\n}\n#endif\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return fmin(x, y);\n}\nstatic inline float fmax32(float x, float y)\n{\n    return fmax(x, y);\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline bool cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return (float) x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return (float) x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return (float) x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint",
            "32_t fptoui_f32_i32(float x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return (uint64_t) x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_cosh32(float x)\n{\n    return cosh(x);\n}\nstatic inline float futrts_sinh32(float x)\n{\n    return sinh(x);\n}\nstatic inline float futrts_tanh32(float x)\n{\n    return tanh(x);\n}\nstatic inline float futrts_acosh32(float x)\n{\n    return acosh(x);\n}\nstatic inline float futrts_asinh32(float x)\n{\n    return asinh(x);\n}\nstatic inline float futrts_atanh32(float x)\n{\n    return atanh(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_gamma32(float x)\n{\n    return tgamma(x);\n}\nstatic inline float futrts_lgamma32(float x)\n{\n    return lgamma(x);\n}\nstatic inline bool futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#ifdef __OPENCL_VERSION__\nstatic inline float fmod32(float x, float y)\n{\n    ret",
            "urn fmod(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floor(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceil(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return mad(a, b, c);\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fma(a, b, c);\n}\n#else\nstatic inline float fmod32(float x, float y)\n{\n    return fmodf(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rintf(x);\n}\nstatic inline float futrts_floor32(float x)\n{\n    return floorf(x);\n}\nstatic inline float futrts_ceil32(float x)\n{\n    return ceilf(x);\n}\nstatic inline float futrts_lerp32(float v0, float v1, float t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline float futrts_mad32(float a, float b, float c)\n{\n    return a * b + c;\n}\nstatic inline float futrts_fma32(float a, float b, float c)\n{\n    return fmaf(a, b, c);\n}\n#endif\nstatic inline double fdiv64(double x, double y)\n{\n    return x / y;\n}\nstatic inline double fadd64(double x, double y)\n{\n    return x + y;\n}\nstatic inline double fsub64(double x, double y)\n{\n    return x - y;\n}\nstatic inline double fmul64(double x, double y)\n{\n    return x * y;\n}\nstatic inline double fmin64(double x, double y)\n{\n    return fmin(x, y);\n}\nstatic inline double fmax64(double x, double y)\n{\n    return fmax(x, y);\n}\nstatic inline double fpow64(double x, double y)\n{\n    return pow(x, y);\n}\nstatic inline bool cmplt64(double x, double y)\n{\n    return x < y;\n}\nstatic inline bool cmple64(double x, double y)\n{\n    return x <= y;\n}\nstatic inline double sitofp_i8_f64(int8_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i16_f64(int16_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i32_f64(int32_t x)\n{\n    return (double) x;\n}\nstatic inline double sitofp_i64_f64(int64_t x)\n{\n    return (double) x;\n}\nst",
            "atic inline double uitofp_i8_f64(uint8_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i16_f64(uint16_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i32_f64(uint32_t x)\n{\n    return (double) x;\n}\nstatic inline double uitofp_i64_f64(uint64_t x)\n{\n    return (double) x;\n}\nstatic inline int8_t fptosi_f64_i8(double x)\n{\n    return (int8_t) x;\n}\nstatic inline int16_t fptosi_f64_i16(double x)\n{\n    return (int16_t) x;\n}\nstatic inline int32_t fptosi_f64_i32(double x)\n{\n    return (int32_t) x;\n}\nstatic inline int64_t fptosi_f64_i64(double x)\n{\n    return (int64_t) x;\n}\nstatic inline uint8_t fptoui_f64_i8(double x)\n{\n    return (uint8_t) x;\n}\nstatic inline uint16_t fptoui_f64_i16(double x)\n{\n    return (uint16_t) x;\n}\nstatic inline uint32_t fptoui_f64_i32(double x)\n{\n    return (uint32_t) x;\n}\nstatic inline uint64_t fptoui_f64_i64(double x)\n{\n    return (uint64_t) x;\n}\nstatic inline double futrts_log64(double x)\n{\n    return log(x);\n}\nstatic inline double futrts_log2_64(double x)\n{\n    return log2(x);\n}\nstatic inline double futrts_log10_64(double x)\n{\n    return log10(x);\n}\nstatic inline double futrts_sqrt64(double x)\n{\n    return sqrt(x);\n}\nstatic inline double futrts_exp64(double x)\n{\n    return exp(x);\n}\nstatic inline double futrts_cos64(double x)\n{\n    return cos(x);\n}\nstatic inline double futrts_sin64(double x)\n{\n    return sin(x);\n}\nstatic inline double futrts_tan64(double x)\n{\n    return tan(x);\n}\nstatic inline double futrts_acos64(double x)\n{\n    return acos(x);\n}\nstatic inline double futrts_asin64(double x)\n{\n    return asin(x);\n}\nstatic inline double futrts_atan64(double x)\n{\n    return atan(x);\n}\nstatic inline double futrts_cosh64(double x)\n{\n    return cosh(x);\n}\nstatic inline double futrts_sinh64(double x)\n{\n    return sinh(x);\n}\nstatic inline double futrts_tanh64(double x)\n{\n    return tanh(x);\n}\nstatic inline double futrts_acosh64(double x)\n{\n    return acosh(x);\n}\nstatic inline double futrts_asinh64(double x)\n{\n    return asinh(x);\n}\n",
            "static inline double futrts_atanh64(double x)\n{\n    return atanh(x);\n}\nstatic inline double futrts_atan2_64(double x, double y)\n{\n    return atan2(x, y);\n}\nstatic inline double futrts_gamma64(double x)\n{\n    return tgamma(x);\n}\nstatic inline double futrts_lgamma64(double x)\n{\n    return lgamma(x);\n}\nstatic inline double futrts_fma64(double a, double b, double c)\n{\n    return fma(a, b, c);\n}\nstatic inline double futrts_round64(double x)\n{\n    return rint(x);\n}\nstatic inline double futrts_ceil64(double x)\n{\n    return ceil(x);\n}\nstatic inline double futrts_floor64(double x)\n{\n    return floor(x);\n}\nstatic inline bool futrts_isnan64(double x)\n{\n    return isnan(x);\n}\nstatic inline bool futrts_isinf64(double x)\n{\n    return isinf(x);\n}\nstatic inline int64_t futrts_to_bits64(double x)\n{\n    union {\n        double f;\n        int64_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double futrts_from_bits64(int64_t x)\n{\n    union {\n        int64_t f;\n        double t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline double fmod64(double x, double y)\n{\n    return fmod(x, y);\n}\n#ifdef __OPENCL_VERSION__\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return mix(v0, v1, t);\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return mad(a, b, c);\n}\n#else\nstatic inline double futrts_lerp64(double v0, double v1, double t)\n{\n    return v0 + (v1 - v0) * t;\n}\nstatic inline double futrts_mad64(double a, double b, double c)\n{\n    return a * b + c;\n}\n#endif\nstatic inline float fpconv_f32_f32(float x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f32_f64(float x)\n{\n    return (double) x;\n}\nstatic inline float fpconv_f64_f32(double x)\n{\n    return (float) x;\n}\nstatic inline double fpconv_f64_f64(double x)\n{\n    return (double) x;\n}\n// Start of atomics.h\n\ninline int32_t atomic_add_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_ad",
            "d(p, x);\n#endif\n}\n\ninline int32_t atomic_add_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((int32_t*)p, x);\n#else\n  return atomic_add(p, x);\n#endif\n}\n\ninline float atomic_fadd_f32_global(volatile __global float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __global int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline float atomic_fadd_f32_local(volatile __local float *p, float x) {\n#ifdef FUTHARK_CUDA\n  return atomicAdd((float*)p, x);\n#else\n  union { int32_t i; float f; } old;\n  union { int32_t i; float f; } assumed;\n  old.f = *p;\n  do {\n    assumed.f = old.f;\n    old.f = old.f + x;\n    old.i = atomic_cmpxchg((volatile __local int32_t*)p, assumed.i, old.i);\n  } while (assumed.i != old.i);\n  return old.f;\n#endif\n}\n\ninline int32_t atomic_smax_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smax_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((int32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_smin_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((int32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umax_i32_local(volatile __local uint32_t *p, uint32_t ",
            "x) {\n#ifdef FUTHARK_CUDA\n  return atomicMax((uint32_t*)p, x);\n#else\n  return atomic_max(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_global(volatile __global uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline uint32_t atomic_umin_i32_local(volatile __local uint32_t *p, uint32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicMin((uint32_t*)p, x);\n#else\n  return atomic_min(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_and_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicAnd((int32_t*)p, x);\n#else\n  return atomic_and(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_or_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicOr((int32_t*)p, x);\n#else\n  return atomic_or(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xor_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicXor((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_global(volatile __global int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_xchg_i32_local(volatile __local int32_t *p, int32_t x) {\n#ifdef FUTHARK_CUDA\n  return atomicExch((int32_t*)p, x);\n#else\n  return atomic_xor(p, x);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_global(volatile __global int32_t *p,\n                                    ",
            "     int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\ninline int32_t atomic_cmpxchg_i32_local(volatile __local int32_t *p,\n                                         int32_t cmp, int32_t val) {\n#ifdef FUTHARK_CUDA\n  return atomicCAS((int32_t*)p, cmp, val);\n#else\n  return atomic_cmpxchg(p, cmp, val);\n#endif\n}\n\n// End of atomics.h\n\n__kernel void iota_2225(int32_t res_943, __global unsigned char *mem_1923)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t iota_gtid_2225;\n    int32_t iota_ltid_2226;\n    int32_t iota_gid_2227;\n    \n    iota_gtid_2225 = get_global_id(0);\n    iota_ltid_2226 = get_local_id(0);\n    iota_gid_2227 = get_group_id(0);\n    if (slt32(iota_gtid_2225, res_943)) {\n        ((__global int32_t *) mem_1923)[iota_gtid_2225] =\n            sext_i32_i32(iota_gtid_2225);\n    }\n    \n  error_0:\n    return;\n}\n__kernel void iota_2266(int32_t pts_per_node_at_lev_1013, __global\n                        unsigned char *mem_1949)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t iota_gtid_2266;\n    int32_t iota_ltid_2267;\n    int32_t iota_gid_2268;\n    \n    iota_gtid_2266 = get_global_id(0);\n    iota_ltid_2267 = get_local_id(0);\n    iota_gid_2268 = get_group_id(0);\n    if (slt32(iota_gtid_2266, pts_per_node_at_lev_1013)) {\n        ((__global int32_t *) mem_1949)[iota_gtid_2266] =\n            sext_i32_i32(iota_gtid_2266);\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_f32(const int block_dim0, const int block_dim1,\n                                const int block_dim2,\n                                uint block_11_backing_offset_0,\n                                int32_t destoffset_1, int32_t srcoffset_3,\n                                int32_t num_arrays_4, int32_t x_elems_5,\n                                int32_t y_elems_6, int32_t in_elems_7,\n         ",
            "                       int32_t out_elems_8, int32_t mulx_9,\n                                int32_t muly_10, __global\n                                unsigned char *destmem_0, __global\n                                unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 32), get_local_id_1_39);\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 = add32(mul32(add32(y_index_32, mul32(j_43, 8)),\n                                              x_elems_5), x_index_31);\n            \n            if (slt32(add32(y_index_32, mul32(j_43, 8)), y_elems_6) &&\n                slt32(index_in_35, in_elems_7)) {\n                ((__local\n                  float *) block_11)[add32(mul32(add32(get_local_id_1_39,\n                                                       mul32(j_43, 8)), 33),\n                                      ",
            "     get_local_id_0_38)] = ((__global\n                                                                   float *) srcmem_2)[add32(idata_offset_34,\n                                                                                            index_in_35)];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(mul32(get_group_id_1_41, 32), get_local_id_0_38);\n    y_index_32 = add32(mul32(get_group_id_0_40, 32), get_local_id_1_39);\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = add32(mul32(add32(y_index_32, mul32(j_43,\n                                                                       8)),\n                                               y_elems_6), x_index_31);\n            \n            if (slt32(add32(y_index_32, mul32(j_43, 8)), x_elems_5) &&\n                slt32(index_out_36, out_elems_8)) {\n                ((__global float *) destmem_0)[add32(odata_offset_33,\n                                                     index_out_36)] = ((__local\n                                                                        float *) block_11)[add32(add32(mul32(get_local_id_0_38,\n                                                                                                             33),\n                                                                                                       get_local_id_1_39),\n                                                                                                 mul32(j_43,\n                                                                                                       8))];\n            }\n        }\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_f32_low_height(const int block_dim0, const\n                                           int block_dim1, const int block_dim2,\n                                           uint block_11_backing_offset_0,\n                                           int32_t destoffset_1,",
            "\n                                           int32_t srcoffset_3,\n                                           int32_t num_arrays_4,\n                                           int32_t x_elems_5, int32_t y_elems_6,\n                                           int32_t in_elems_7,\n                                           int32_t out_elems_8, int32_t mulx_9,\n                                           int32_t muly_10, __global\n                                           unsigned char *destmem_0, __global\n                                           unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = add32(add32(mul32(mul32(get_group_id_0_40, 16),\n                                           mulx_9), get_local_id_0_38),\n                               mul32(srem32(get_local_id_1_39, mulx_9), 16));\n    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 16),\n                               squot32(get_local_id_1_39, mulx_9));\n    int32_t index_in_35 = add32(mul32(y_inde",
            "x_32, x_elems_5), x_index_31);\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local float *) block_11)[add32(mul32(get_local_id_1_39, 17),\n                                           get_local_id_0_38)] = ((__global\n                                                                   float *) srcmem_2)[add32(idata_offset_34,\n                                                                                            index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(mul32(get_group_id_1_41, 16), squot32(get_local_id_0_38,\n                                                             mulx_9));\n    y_index_32 = add32(add32(mul32(mul32(get_group_id_0_40, 16), mulx_9),\n                             get_local_id_1_39), mul32(srem32(get_local_id_0_38,\n                                                              mulx_9), 16));\n    \n    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global float *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__local float *) block_11)[add32(mul32(get_local_id_0_38, 17),\n                                               get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_f32_low_width(const int block_dim0, const\n                                          int block_dim1, const int block_dim2,\n                                          uint block_11_backing_offset_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                   ",
            "       int32_t in_elems_7,\n                                          int32_t out_elems_8, int32_t mulx_9,\n                                          int32_t muly_10, __global\n                                          unsigned char *destmem_0, __global\n                                          unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = add32(mul32(get_group_id_0_40, 16),\n                               squot32(get_local_id_0_38, muly_10));\n    int32_t y_index_32 = add32(add32(mul32(mul32(get_group_id_1_41, 16),\n                                           muly_10), get_local_id_1_39),\n                               mul32(srem32(get_local_id_0_38, muly_10), 16));\n    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local float *) block_11)[add32(mul32(get_local_id_1",
            "_39, 17),\n                                           get_local_id_0_38)] = ((__global\n                                                                   float *) srcmem_2)[add32(idata_offset_34,\n                                                                                            index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(add32(mul32(mul32(get_group_id_1_41, 16), muly_10),\n                             get_local_id_0_38), mul32(srem32(get_local_id_1_39,\n                                                              muly_10), 16));\n    y_index_32 = add32(mul32(get_group_id_0_40, 16), squot32(get_local_id_1_39,\n                                                             muly_10));\n    \n    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global float *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__local float *) block_11)[add32(mul32(get_local_id_0_38, 17),\n                                               get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_f32_small(uint block_11_backing_offset_0,\n                                      int32_t destoffset_1, int32_t srcoffset_3,\n                                      int32_t num_arrays_4, int32_t x_elems_5,\n                                      int32_t y_elems_6, int32_t in_elems_7,\n                                      int32_t out_elems_8, int32_t mulx_9,\n                                      int32_t muly_10, __global\n                                      unsigned char *destmem_0, __global\n                                      unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *blo",
            "ck_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(squot32(get_global_id_0_37,\n                                                mul32(y_elems_6, x_elems_5)),\n                                        mul32(y_elems_6, x_elems_5));\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, mul32(y_elems_6,\n                                                                  x_elems_5)),\n                                 y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);\n    int32_t index_out_36 = add32(mul32(x_index_31, y_elems_6), y_index_32);\n    \n    if (slt32(get_global_id_0_37, in_elems_7)) {\n        ((__global float *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__global float *) srcmem_2)[add32(idata_offset_34, index_in_35)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_i32(const int block_dim0, const int block_dim1,\n                                const int block_dim2,\n                                uint block_11_backing_offset_0,\n                                int32_t destoffset_1, int32_t srcoffset_3,\n                                int32_t num_arr",
            "ays_4, int32_t x_elems_5,\n                                int32_t y_elems_6, int32_t in_elems_7,\n                                int32_t out_elems_8, int32_t mulx_9,\n                                int32_t muly_10, __global\n                                unsigned char *destmem_0, __global\n                                unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = get_global_id_0_37;\n    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 32), get_local_id_1_39);\n    \n    if (slt32(x_index_31, x_elems_5)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_in_35 = add32(mul32(add32(y_index_32, mul32(j_43, 8)),\n                                              x_elems_5), x_index_31);\n            \n            if (slt32(add32(y_index_32, mul32(j_43, 8)), y_elems_6) &&\n                slt32(index_in_35, in_elems_7)) {\n                ((__local\n                  int32_t *) block_11)[add32(mul32(add32(get_local_id_1_39,\n       ",
            "                                                  mul32(j_43, 8)), 33),\n                                             get_local_id_0_38)] = ((__global\n                                                                     int32_t *) srcmem_2)[add32(idata_offset_34,\n                                                                                                index_in_35)];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(mul32(get_group_id_1_41, 32), get_local_id_0_38);\n    y_index_32 = add32(mul32(get_group_id_0_40, 32), get_local_id_1_39);\n    if (slt32(x_index_31, y_elems_6)) {\n        for (int32_t j_43 = 0; j_43 < 4; j_43++) {\n            int32_t index_out_36 = add32(mul32(add32(y_index_32, mul32(j_43,\n                                                                       8)),\n                                               y_elems_6), x_index_31);\n            \n            if (slt32(add32(y_index_32, mul32(j_43, 8)), x_elems_5) &&\n                slt32(index_out_36, out_elems_8)) {\n                ((__global int32_t *) destmem_0)[add32(odata_offset_33,\n                                                       index_out_36)] =\n                    ((__local\n                      int32_t *) block_11)[add32(add32(mul32(get_local_id_0_38,\n                                                             33),\n                                                       get_local_id_1_39),\n                                                 mul32(j_43, 8))];\n            }\n        }\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_i32_low_height(const int block_dim0, const\n                                           int block_dim1, const int block_dim2,\n                                           uint block_11_backing_offset_0,\n                                           int32_t destoffset_1,\n                                           int32_t srcoffset_3,\n                                           int32_t num_arrays_4,\n                     ",
            "                      int32_t x_elems_5, int32_t y_elems_6,\n                                           int32_t in_elems_7,\n                                           int32_t out_elems_8, int32_t mulx_9,\n                                           int32_t muly_10, __global\n                                           unsigned char *destmem_0, __global\n                                           unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = add32(add32(mul32(mul32(get_group_id_0_40, 16),\n                                           mulx_9), get_local_id_0_38),\n                               mul32(srem32(get_local_id_1_39, mulx_9), 16));\n    int32_t y_index_32 = add32(mul32(get_group_id_1_41, 16),\n                               squot32(get_local_id_1_39, mulx_9));\n    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         s",
            "lt32(index_in_35, in_elems_7))) {\n        ((__local int32_t *) block_11)[add32(mul32(get_local_id_1_39, 17),\n                                             get_local_id_0_38)] = ((__global\n                                                                     int32_t *) srcmem_2)[add32(idata_offset_34,\n                                                                                                index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(mul32(get_group_id_1_41, 16), squot32(get_local_id_0_38,\n                                                             mulx_9));\n    y_index_32 = add32(add32(mul32(mul32(get_group_id_0_40, 16), mulx_9),\n                             get_local_id_1_39), mul32(srem32(get_local_id_0_38,\n                                                              mulx_9), 16));\n    \n    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global int32_t *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__local int32_t *) block_11)[add32(mul32(get_local_id_0_38, 17),\n                                                 get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_i32_low_width(const int block_dim0, const\n                                          int block_dim1, const int block_dim2,\n                                          uint block_11_backing_offset_0,\n                                          int32_t destoffset_1,\n                                          int32_t srcoffset_3,\n                                          int32_t num_arrays_4,\n                                          int32_t x_elems_5, int32_t y_elems_6,\n                                          int32_t in_elems_7,\n                                          int32_t out_elems_8, int32_t mulx_9,\n                           ",
            "               int32_t muly_10, __global\n                                          unsigned char *destmem_0, __global\n                                          unsigned char *srcmem_2)\n{\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_global_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(mul32(get_group_id_2_42, x_elems_5),\n                                        y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t x_index_31 = add32(mul32(get_group_id_0_40, 16),\n                               squot32(get_local_id_0_38, muly_10));\n    int32_t y_index_32 = add32(add32(mul32(mul32(get_group_id_1_41, 16),\n                                           muly_10), get_local_id_1_39),\n                               mul32(srem32(get_local_id_0_38, muly_10), 16));\n    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);\n    \n    if (slt32(x_index_31, x_elems_5) && (slt32(y_index_32, y_elems_6) &&\n                                         slt32(index_in_35, in_elems_7))) {\n        ((__local int32_t *) block_11)[add32(mul32(get_local_id_1_39, 17),\n                                             get_local_id_0_38)] = ((__global\n                                           ",
            "                          int32_t *) srcmem_2)[add32(idata_offset_34,\n                                                                                                index_in_35)];\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    x_index_31 = add32(add32(mul32(mul32(get_group_id_1_41, 16), muly_10),\n                             get_local_id_0_38), mul32(srem32(get_local_id_1_39,\n                                                              muly_10), 16));\n    y_index_32 = add32(mul32(get_group_id_0_40, 16), squot32(get_local_id_1_39,\n                                                             muly_10));\n    \n    int32_t index_out_36 = add32(mul32(y_index_32, y_elems_6), x_index_31);\n    \n    if (slt32(x_index_31, y_elems_6) && (slt32(y_index_32, x_elems_5) &&\n                                         slt32(index_out_36, out_elems_8))) {\n        ((__global int32_t *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__local int32_t *) block_11)[add32(mul32(get_local_id_0_38, 17),\n                                                 get_local_id_1_39)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void map_transpose_i32_small(uint block_11_backing_offset_0,\n                                      int32_t destoffset_1, int32_t srcoffset_3,\n                                      int32_t num_arrays_4, int32_t x_elems_5,\n                                      int32_t y_elems_6, int32_t in_elems_7,\n                                      int32_t out_elems_8, int32_t mulx_9,\n                                      int32_t muly_10, __global\n                                      unsigned char *destmem_0, __global\n                                      unsigned char *srcmem_2)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *block_11_backing_0 = &shared_mem[block_11_backing_offset_0];\n    __local char *block_11;\n    \n    block_11 = (__local char *) block_11_backing_0;\n    \n    int32_t get_global_id_0_37;\n    \n    get_g",
            "lobal_id_0_37 = get_global_id(0);\n    \n    int32_t get_local_id_0_38;\n    \n    get_local_id_0_38 = get_local_id(0);\n    \n    int32_t get_local_id_1_39;\n    \n    get_local_id_1_39 = get_local_id(1);\n    \n    int32_t get_group_id_0_40;\n    \n    get_group_id_0_40 = get_group_id(0);\n    \n    int32_t get_group_id_1_41;\n    \n    get_group_id_1_41 = get_group_id(1);\n    \n    int32_t get_group_id_2_42;\n    \n    get_group_id_2_42 = get_group_id(2);\n    \n    int32_t our_array_offset_30 = mul32(squot32(get_global_id_0_37,\n                                                mul32(y_elems_6, x_elems_5)),\n                                        mul32(y_elems_6, x_elems_5));\n    int32_t x_index_31 = squot32(srem32(get_global_id_0_37, mul32(y_elems_6,\n                                                                  x_elems_5)),\n                                 y_elems_6);\n    int32_t y_index_32 = srem32(get_global_id_0_37, y_elems_6);\n    int32_t odata_offset_33 = add32(squot32(destoffset_1, 4),\n                                    our_array_offset_30);\n    int32_t idata_offset_34 = add32(squot32(srcoffset_3, 4),\n                                    our_array_offset_30);\n    int32_t index_in_35 = add32(mul32(y_index_32, x_elems_5), x_index_31);\n    int32_t index_out_36 = add32(mul32(x_index_31, y_elems_6), y_index_32);\n    \n    if (slt32(get_global_id_0_37, in_elems_7)) {\n        ((__global int32_t *) destmem_0)[add32(odata_offset_33, index_out_36)] =\n            ((__global int32_t *) srcmem_2)[add32(idata_offset_34,\n                                                  index_in_35)];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void replicate_2234(__global unsigned char *mem_2230,\n                             int32_t num_elems_2231, float val_2232)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_2234;\n    int32_t replicate_ltid_2235;\n    int32_t replicate_gid_2236;\n    \n    replicate_gtid_2234 = get_global_id(0);\n    re",
            "plicate_ltid_2235 = get_local_id(0);\n    replicate_gid_2236 = get_group_id(0);\n    if (slt32(replicate_gtid_2234, num_elems_2231)) {\n        ((__global float *) mem_2230)[replicate_gtid_2234] = val_2232;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void replicate_2243(__global unsigned char *mem_2239,\n                             int32_t num_elems_2240, int32_t val_2241)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_2243;\n    int32_t replicate_ltid_2244;\n    int32_t replicate_gid_2245;\n    \n    replicate_gtid_2243 = get_global_id(0);\n    replicate_ltid_2244 = get_local_id(0);\n    replicate_gid_2245 = get_group_id(0);\n    if (slt32(replicate_gtid_2243, num_elems_2240)) {\n        ((__global int32_t *) mem_2239)[replicate_gtid_2243] = val_2241;\n    }\n    \n  error_0:\n    return;\n}\n__kernel void replicate_2281(int32_t nodes_this_lvl_1009,\n                             int32_t pts_per_node_at_lev_1013, __global\n                             unsigned char *mem_1949, __global\n                             unsigned char *mem_1963)\n{\n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    int32_t replicate_gtid_2281;\n    int32_t replicate_ltid_2282;\n    int32_t replicate_gid_2283;\n    \n    replicate_gtid_2281 = get_global_id(0);\n    replicate_ltid_2282 = get_local_id(0);\n    replicate_gid_2283 = get_group_id(0);\n    if (slt32(replicate_gtid_2281, mul32(nodes_this_lvl_1009,\n                                         pts_per_node_at_lev_1013))) {\n        ((__global int32_t *) mem_1963)[add32(mul32(squot32(replicate_gtid_2281,\n                                                            pts_per_node_at_lev_1013),\n                                                    pts_per_node_at_lev_1013),\n                                              sub32(replicate_gtid_2281,\n                                                    mul32(squot32(replicate_gtid_2281,\n                            ",
            "                                      pts_per_node_at_lev_1013),\n                                                          pts_per_node_at_lev_1013)))] =\n            ((__global int32_t *) mem_1949)[sub32(replicate_gtid_2281,\n                                                  mul32(squot32(replicate_gtid_2281,\n                                                                pts_per_node_at_lev_1013),\n                                                        pts_per_node_at_lev_1013))];\n    }\n    \n  error_0:\n    return;\n}\n__kernel void scan_stage1_1689(__global int *global_failure,\n                               uint scan_arr_mem_2311_backing_offset_0,\n                               uint scan_arr_mem_2309_backing_offset_1,\n                               int32_t nodes_this_lvl_1009,\n                               int32_t pts_per_node_at_lev_1013, __global\n                               unsigned char *mem_1971, __global\n                               unsigned char *mem_1977, __global\n                               unsigned char *mem_1982, __global\n                               unsigned char *mem_1987,\n                               int32_t num_threads_2297)\n{\n    #define segscan_group_sizze_1684 (mainzisegscan_group_sizze_1683)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *scan_arr_mem_2311_backing_1 =\n                  &shared_mem[scan_arr_mem_2311_backing_offset_0];\n    volatile char *scan_arr_mem_2309_backing_0 =\n                  &shared_mem[scan_arr_mem_2309_backing_offset_1];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2304;\n    int32_t local_tid_2305;\n    int32_t group_sizze_2308;\n    int32_t wave_sizze_2307;\n    int32_t group_tid_2306;\n    \n    global_tid_2304 = get_global_id(0);\n    local_tid_2305 = get_local_id(0);\n    group_sizze_2308 = get_local_size(0);\n    wave_sizze_2307 = LOCKSTEP_WIDTH;\n    group_tid_2306 = get_group_id(0);\n    \n    int32_t phys_tid_1689 ",
            "= global_tid_2304;\n    __local char *scan_arr_mem_2309;\n    \n    scan_arr_mem_2309 = (__local char *) scan_arr_mem_2309_backing_0;\n    \n    __local char *scan_arr_mem_2311;\n    \n    scan_arr_mem_2311 = (__local char *) scan_arr_mem_2311_backing_1;\n    \n    int32_t x_1690;\n    int32_t x_1691;\n    int32_t x_1692;\n    int32_t x_1693;\n    \n    x_1690 = 0;\n    x_1691 = 0;\n    for (int32_t j_2313 = 0; j_2313 <\n         squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                   pts_per_node_at_lev_1013), num_threads_2297),\n                       1), num_threads_2297); j_2313++) {\n        int32_t chunk_offset_2314 = add32(mul32(segscan_group_sizze_1684,\n                                                j_2313), mul32(group_tid_2306,\n                                                               mul32(segscan_group_sizze_1684,\n                                                                     squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                               pts_per_node_at_lev_1013),\n                                                                                         num_threads_2297),\n                                                                                   1),\n                                                                             num_threads_2297))));\n        int32_t flat_idx_2315 = add32(chunk_offset_2314, local_tid_2305);\n        int32_t gtid_1676 = squot32(flat_idx_2315, pts_per_node_at_lev_1013);\n        int32_t gtid_1688;\n        \n        gtid_1688 = sub32(flat_idx_2315, mul32(squot32(flat_idx_2315,\n                                                       pts_per_node_at_lev_1013),\n                                               pts_per_node_at_lev_1013));\n        // threads in bounds read input; others get neutral element\n        {\n            if (slt32(gtid_1676, nodes_this_lvl_1009) && slt32(gtid_1688,\n                                                  ",
            "             pts_per_node_at_lev_1013)) {\n                int32_t x_1697 = ((__global\n                                   int32_t *) mem_1971)[add32(mul32(gtid_1676,\n                                                                    pts_per_node_at_lev_1013),\n                                                              gtid_1688)];\n                int32_t res_1698 = sub32(1, x_1697);\n                \n                // write to-scan values to parameters\n                {\n                    x_1692 = res_1698;\n                    x_1693 = x_1697;\n                }\n                // write mapped values results to global memory\n                {\n                    ((__global int32_t *) mem_1987)[add32(mul32(gtid_1676,\n                                                                pts_per_node_at_lev_1013),\n                                                          gtid_1688)] =\n                        res_1698;\n                }\n            } else {\n                x_1692 = 0;\n                x_1693 = 0;\n            }\n        }\n        // combine with carry and write to local memory\n        {\n            int32_t res_1694 = add32(x_1690, x_1692);\n            int32_t res_1695 = add32(x_1691, x_1693);\n            \n            ((__local int32_t *) scan_arr_mem_2309)[local_tid_2305] = res_1694;\n            ((__local int32_t *) scan_arr_mem_2311)[local_tid_2305] = res_1695;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        int32_t x_2298;\n        int32_t x_2299;\n        int32_t x_2300;\n        int32_t x_2301;\n        int32_t x_2316;\n        int32_t x_2317;\n        int32_t x_2318;\n        int32_t x_2319;\n        int32_t skip_threads_2322;\n        \n        // read input for in-block scan\n        {\n            if (slt32(local_tid_2305, segscan_group_sizze_1684)) {\n                x_2300 = ((volatile __local\n                           int32_t *) scan_arr_mem_2309)[local_tid_2305];\n                x_2301 = ((volatile __local\n                           int32_t *)",
            " scan_arr_mem_2311)[local_tid_2305];\n                if (sub32(local_tid_2305, mul32(squot32(local_tid_2305, 32),\n                                                32)) == 0) {\n                    x_2298 = x_2300;\n                    x_2299 = x_2301;\n                }\n            }\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_2322 = 1;\n            while (slt32(skip_threads_2322, 32)) {\n                if (sle32(skip_threads_2322, sub32(local_tid_2305,\n                                                   mul32(squot32(local_tid_2305,\n                                                                 32), 32))) &&\n                    slt32(local_tid_2305, segscan_group_sizze_1684)) {\n                    // read operands\n                    {\n                        x_2298 = ((volatile __local\n                                   int32_t *) scan_arr_mem_2309)[sub32(local_tid_2305,\n                                                                       skip_threads_2322)];\n                        x_2299 = ((volatile __local\n                                   int32_t *) scan_arr_mem_2311)[sub32(local_tid_2305,\n                                                                       skip_threads_2322)];\n                    }\n                    // perform operation\n                    {\n                        bool inactive_2323 = slt32(srem32(add32(local_tid_2305,\n                                                                chunk_offset_2314),\n                                                          pts_per_node_at_lev_1013),\n                                                   sub32(add32(local_tid_2305,\n                                                               chunk_offset_2314),\n                                                         add32(sub32(local_tid_2305,\n                                                                     skip_threads_2322),\n                                                               chunk",
            "_offset_2314)));\n                        \n                        if (inactive_2323) {\n                            x_2298 = x_2300;\n                            x_2299 = x_2301;\n                        }\n                        if (!inactive_2323) {\n                            int32_t res_2302 = add32(x_2298, x_2300);\n                            int32_t res_2303 = add32(x_2299, x_2301);\n                            \n                            x_2298 = res_2302;\n                            x_2299 = res_2303;\n                        }\n                    }\n                }\n                if (sle32(wave_sizze_2307, skip_threads_2322)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_2322, sub32(local_tid_2305,\n                                                   mul32(squot32(local_tid_2305,\n                                                                 32), 32))) &&\n                    slt32(local_tid_2305, segscan_group_sizze_1684)) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_2309)[local_tid_2305] =\n                            x_2298;\n                        x_2300 = x_2298;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_2311)[local_tid_2305] =\n                            x_2299;\n                        x_2301 = x_2299;\n                    }\n                }\n                if (sle32(wave_sizze_2307, skip_threads_2322)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_2322 *= 2;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // last thread of block 'i' writes its result to offset 'i'\n        {\n            if (sub32(local_tid_2305, mul32(squot32(local_tid_2305, 32), 32)) ==\n                31 && slt32(local_tid_2305, segscan_group_sizze_1684)) {\n                ((volatile __local\n   ",
            "               int32_t *) scan_arr_mem_2309)[squot32(local_tid_2305, 32)] =\n                    x_2298;\n                ((volatile __local\n                  int32_t *) scan_arr_mem_2311)[squot32(local_tid_2305, 32)] =\n                    x_2299;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n        {\n            int32_t skip_threads_2324;\n            \n            // read input for in-block scan\n            {\n                if (squot32(local_tid_2305, 32) == 0 && slt32(local_tid_2305,\n                                                              segscan_group_sizze_1684)) {\n                    x_2318 = ((volatile __local\n                               int32_t *) scan_arr_mem_2309)[local_tid_2305];\n                    x_2319 = ((volatile __local\n                               int32_t *) scan_arr_mem_2311)[local_tid_2305];\n                    if (sub32(local_tid_2305, mul32(squot32(local_tid_2305, 32),\n                                                    32)) == 0) {\n                        x_2316 = x_2318;\n                        x_2317 = x_2319;\n                    }\n                }\n            }\n            // in-block scan (hopefully no barriers needed)\n            {\n                skip_threads_2324 = 1;\n                while (slt32(skip_threads_2324, 32)) {\n                    if (sle32(skip_threads_2324, sub32(local_tid_2305,\n                                                       mul32(squot32(local_tid_2305,\n                                                                     32),\n                                                             32))) &&\n                        (squot32(local_tid_2305, 32) == 0 &&\n                         slt32(local_tid_2305, segscan_group_sizze_1684))) {\n                        // read operands\n                        {\n                            x_2316 = ((volatile __local\n                                       int32_t *) ",
            "scan_arr_mem_2309)[sub32(local_tid_2305,\n                                                                           skip_threads_2324)];\n                            x_2317 = ((volatile __local\n                                       int32_t *) scan_arr_mem_2311)[sub32(local_tid_2305,\n                                                                           skip_threads_2324)];\n                        }\n                        // perform operation\n                        {\n                            bool inactive_2325 =\n                                 slt32(srem32(add32(sub32(add32(mul32(local_tid_2305,\n                                                                      32), 32),\n                                                          1),\n                                                    chunk_offset_2314),\n                                              pts_per_node_at_lev_1013),\n                                       sub32(add32(sub32(add32(mul32(local_tid_2305,\n                                                                     32), 32),\n                                                         1), chunk_offset_2314),\n                                             add32(sub32(add32(mul32(sub32(local_tid_2305,\n                                                                           skip_threads_2324),\n                                                                     32), 32),\n                                                         1),\n                                                   chunk_offset_2314)));\n                            \n                            if (inactive_2325) {\n                                x_2316 = x_2318;\n                                x_2317 = x_2319;\n                            }\n                            if (!inactive_2325) {\n                                int32_t res_2320 = add32(x_2316, x_2318);\n                                int32_t res_2321 = add32(x_2317, x_2319);\n                                \n                    ",
            "            x_2316 = res_2320;\n                                x_2317 = res_2321;\n                            }\n                        }\n                    }\n                    if (sle32(wave_sizze_2307, skip_threads_2324)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    if (sle32(skip_threads_2324, sub32(local_tid_2305,\n                                                       mul32(squot32(local_tid_2305,\n                                                                     32),\n                                                             32))) &&\n                        (squot32(local_tid_2305, 32) == 0 &&\n                         slt32(local_tid_2305, segscan_group_sizze_1684))) {\n                        // write result\n                        {\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_2309)[local_tid_2305] =\n                                x_2316;\n                            x_2318 = x_2316;\n                            ((volatile __local\n                              int32_t *) scan_arr_mem_2311)[local_tid_2305] =\n                                x_2317;\n                            x_2319 = x_2317;\n                        }\n                    }\n                    if (sle32(wave_sizze_2307, skip_threads_2324)) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                    }\n                    skip_threads_2324 *= 2;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // carry-in for every block except the first\n        {\n            if (!(squot32(local_tid_2305, 32) == 0 || !slt32(local_tid_2305,\n                                                             segscan_group_sizze_1684))) {\n                // read operands\n                {\n                    x_2300 = x_2298;\n                    x_2301 = x_2299;\n                    x_2298 = ((__local\n                               int32_t *) scan_arr_mem_2309",
            ")[sub32(squot32(local_tid_2305,\n                                                                           32),\n                                                                   1)];\n                    x_2299 = ((__local\n                               int32_t *) scan_arr_mem_2311)[sub32(squot32(local_tid_2305,\n                                                                           32),\n                                                                   1)];\n                }\n                // perform operation\n                {\n                    bool inactive_2326 = slt32(srem32(add32(local_tid_2305,\n                                                            chunk_offset_2314),\n                                                      pts_per_node_at_lev_1013),\n                                               sub32(add32(local_tid_2305,\n                                                           chunk_offset_2314),\n                                                     add32(sub32(mul32(squot32(local_tid_2305,\n                                                                               32),\n                                                                       32), 1),\n                                                           chunk_offset_2314)));\n                    \n                    if (inactive_2326) {\n                        x_2298 = x_2300;\n                        x_2299 = x_2301;\n                    }\n                    if (!inactive_2326) {\n                        int32_t res_2302 = add32(x_2298, x_2300);\n                        int32_t res_2303 = add32(x_2299, x_2301);\n                        \n                        x_2298 = res_2302;\n                        x_2299 = res_2303;\n                    }\n                }\n                // write final result\n                {\n                    ((__local int32_t *) scan_arr_mem_2309)[local_tid_2305] =\n                        x_2298;\n                    ((__local int32_t *) scan_arr_mem_2311)[local_tid_230",
            "5] =\n                        x_2299;\n                }\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // restore correct values for first block\n        {\n            if (squot32(local_tid_2305, 32) == 0) {\n                ((__local int32_t *) scan_arr_mem_2309)[local_tid_2305] =\n                    x_2300;\n                ((__local int32_t *) scan_arr_mem_2311)[local_tid_2305] =\n                    x_2301;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // threads in bounds write partial scan result\n        {\n            if (slt32(gtid_1676, nodes_this_lvl_1009) && slt32(gtid_1688,\n                                                               pts_per_node_at_lev_1013)) {\n                ((__global int32_t *) mem_1977)[add32(mul32(gtid_1676,\n                                                            pts_per_node_at_lev_1013),\n                                                      gtid_1688)] = ((__local\n                                                                      int32_t *) scan_arr_mem_2309)[local_tid_2305];\n                ((__global int32_t *) mem_1982)[add32(mul32(gtid_1676,\n                                                            pts_per_node_at_lev_1013),\n                                                      gtid_1688)] = ((__local\n                                                                      int32_t *) scan_arr_mem_2311)[local_tid_2305];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread reads last element as carry-in for next iteration\n        {\n            bool crosses_segment_2327 = slt32(srem32(add32(chunk_offset_2314,\n                                                           segscan_group_sizze_1684),\n                                                     pts_per_node_at_lev_1013),\n                                              sub32(add32(chunk_offset_2314,\n                                                          segscan_group_sizze_1684),\n                ",
            "                                    sub32(add32(chunk_offset_2314,\n                                                                segscan_group_sizze_1684),\n                                                          1)));\n            bool should_load_carry_2328 = local_tid_2305 == 0 &&\n                 !crosses_segment_2327;\n            \n            if (should_load_carry_2328) {\n                x_1690 = ((__local\n                           int32_t *) scan_arr_mem_2309)[sub32(segscan_group_sizze_1684,\n                                                               1)];\n                x_1691 = ((__local\n                           int32_t *) scan_arr_mem_2311)[sub32(segscan_group_sizze_1684,\n                                                               1)];\n            }\n            if (!should_load_carry_2328) {\n                x_1690 = 0;\n                x_1691 = 0;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n    \n  error_1:\n    return;\n    #undef segscan_group_sizze_1684\n}\n__kernel void scan_stage2_1689(__global int *global_failure,\n                               uint scan_arr_mem_2348_backing_offset_0,\n                               uint scan_arr_mem_2346_backing_offset_1,\n                               int32_t nodes_this_lvl_1009,\n                               int32_t pts_per_node_at_lev_1013,\n                               int32_t num_groups_1686, __global\n                               unsigned char *mem_1977, __global\n                               unsigned char *mem_1982,\n                               int32_t num_threads_2297)\n{\n    #define segscan_group_sizze_1684 (mainzisegscan_group_sizze_1683)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *scan_arr_mem_2348_backing_1 =\n                  &shared_mem[scan_arr_mem_2348_backing_offset_0];\n    volatile char *scan_arr_mem_2346_backing_0 =\n                  &shared_mem[scan_arr_mem_2346_backing_offset_1];\n    \n    if (*g",
            "lobal_failure >= 0)\n        return;\n    \n    int32_t global_tid_2341;\n    int32_t local_tid_2342;\n    int32_t group_sizze_2345;\n    int32_t wave_sizze_2344;\n    int32_t group_tid_2343;\n    \n    global_tid_2341 = get_global_id(0);\n    local_tid_2342 = get_local_id(0);\n    group_sizze_2345 = get_local_size(0);\n    wave_sizze_2344 = LOCKSTEP_WIDTH;\n    group_tid_2343 = get_group_id(0);\n    \n    int32_t phys_tid_1689 = global_tid_2341;\n    __local char *scan_arr_mem_2346;\n    \n    scan_arr_mem_2346 = (__local char *) scan_arr_mem_2346_backing_0;\n    \n    __local char *scan_arr_mem_2348;\n    \n    scan_arr_mem_2348 = (__local char *) scan_arr_mem_2348_backing_1;\n    \n    int32_t flat_idx_2350 = sub32(mul32(add32(local_tid_2342, 1),\n                                        mul32(segscan_group_sizze_1684,\n                                              squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                        pts_per_node_at_lev_1013),\n                                                                  num_threads_2297),\n                                                            1),\n                                                      num_threads_2297))), 1);\n    int32_t gtid_1676 = squot32(flat_idx_2350, pts_per_node_at_lev_1013);\n    int32_t gtid_1688;\n    \n    gtid_1688 = sub32(flat_idx_2350, mul32(squot32(flat_idx_2350,\n                                                   pts_per_node_at_lev_1013),\n                                           pts_per_node_at_lev_1013));\n    // threads in bound read carries; others get neutral element\n    {\n        if (slt32(gtid_1676, nodes_this_lvl_1009) && slt32(gtid_1688,\n                                                           pts_per_node_at_lev_1013)) {\n            ((__local int32_t *) scan_arr_mem_2346)[local_tid_2342] = ((__global\n                                                                        int32_t *) mem_1977)[add32(mul32(gtid_1676,\n                             ",
            "                                                                            pts_per_node_at_lev_1013),\n                                                                                                   gtid_1688)];\n            ((__local int32_t *) scan_arr_mem_2348)[local_tid_2342] = ((__global\n                                                                        int32_t *) mem_1982)[add32(mul32(gtid_1676,\n                                                                                                         pts_per_node_at_lev_1013),\n                                                                                                   gtid_1688)];\n        } else {\n            ((__local int32_t *) scan_arr_mem_2346)[local_tid_2342] = 0;\n            ((__local int32_t *) scan_arr_mem_2348)[local_tid_2342] = 0;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t x_2329;\n    int32_t x_2330;\n    int32_t x_2331;\n    int32_t x_2332;\n    int32_t x_2351;\n    int32_t x_2352;\n    int32_t x_2353;\n    int32_t x_2354;\n    int32_t skip_threads_2357;\n    \n    // read input for in-block scan\n    {\n        if (slt32(local_tid_2342, num_groups_1686)) {\n            x_2331 = ((volatile __local\n                       int32_t *) scan_arr_mem_2346)[local_tid_2342];\n            x_2332 = ((volatile __local\n                       int32_t *) scan_arr_mem_2348)[local_tid_2342];\n            if (sub32(local_tid_2342, mul32(squot32(local_tid_2342, 32), 32)) ==\n                0) {\n                x_2329 = x_2331;\n                x_2330 = x_2332;\n            }\n        }\n    }\n    // in-block scan (hopefully no barriers needed)\n    {\n        skip_threads_2357 = 1;\n        while (slt32(skip_threads_2357, 32)) {\n            if (sle32(skip_threads_2357, sub32(local_tid_2342,\n                                               mul32(squot32(local_tid_2342,\n                                                             32), 32))) &&\n                slt32(local_tid_2342, num_groups_1686)) {\n       ",
            "         // read operands\n                {\n                    x_2329 = ((volatile __local\n                               int32_t *) scan_arr_mem_2346)[sub32(local_tid_2342,\n                                                                   skip_threads_2357)];\n                    x_2330 = ((volatile __local\n                               int32_t *) scan_arr_mem_2348)[sub32(local_tid_2342,\n                                                                   skip_threads_2357)];\n                }\n                // perform operation\n                {\n                    bool inactive_2358 =\n                         slt32(srem32(sub32(mul32(add32(local_tid_2342, 1),\n                                                  mul32(segscan_group_sizze_1684,\n                                                        squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                  pts_per_node_at_lev_1013),\n                                                                            num_threads_2297),\n                                                                      1),\n                                                                num_threads_2297))),\n                                            1), pts_per_node_at_lev_1013),\n                               sub32(sub32(mul32(add32(local_tid_2342, 1),\n                                                 mul32(segscan_group_sizze_1684,\n                                                       squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                 pts_per_node_at_lev_1013),\n                                                                           num_threads_2297),\n                                                                     1),\n                                                               num_threads_2297))),\n                                           1),\n                                     sub3",
            "2(mul32(add32(sub32(local_tid_2342,\n                                                             skip_threads_2357),\n                                                       1),\n                                                 mul32(segscan_group_sizze_1684,\n                                                       squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                 pts_per_node_at_lev_1013),\n                                                                           num_threads_2297),\n                                                                     1),\n                                                               num_threads_2297))),\n                                           1)));\n                    \n                    if (inactive_2358) {\n                        x_2329 = x_2331;\n                        x_2330 = x_2332;\n                    }\n                    if (!inactive_2358) {\n                        int32_t res_2333 = add32(x_2329, x_2331);\n                        int32_t res_2334 = add32(x_2330, x_2332);\n                        \n                        x_2329 = res_2333;\n                        x_2330 = res_2334;\n                    }\n                }\n            }\n            if (sle32(wave_sizze_2344, skip_threads_2357)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            if (sle32(skip_threads_2357, sub32(local_tid_2342,\n                                               mul32(squot32(local_tid_2342,\n                                                             32), 32))) &&\n                slt32(local_tid_2342, num_groups_1686)) {\n                // write result\n                {\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_2346)[local_tid_2342] = x_2329;\n                    x_2331 = x_2329;\n                    ((volatile __local\n                      int32_t *) scan_arr_mem_2348)[local_tid_2342] = x_2330;\n             ",
            "       x_2332 = x_2330;\n                }\n            }\n            if (sle32(wave_sizze_2344, skip_threads_2357)) {\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n            skip_threads_2357 *= 2;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // last thread of block 'i' writes its result to offset 'i'\n    {\n        if (sub32(local_tid_2342, mul32(squot32(local_tid_2342, 32), 32)) ==\n            31 && slt32(local_tid_2342, num_groups_1686)) {\n            ((volatile __local\n              int32_t *) scan_arr_mem_2346)[squot32(local_tid_2342, 32)] =\n                x_2329;\n            ((volatile __local\n              int32_t *) scan_arr_mem_2348)[squot32(local_tid_2342, 32)] =\n                x_2330;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n    {\n        int32_t skip_threads_2359;\n        \n        // read input for in-block scan\n        {\n            if (squot32(local_tid_2342, 32) == 0 && slt32(local_tid_2342,\n                                                          num_groups_1686)) {\n                x_2353 = ((volatile __local\n                           int32_t *) scan_arr_mem_2346)[local_tid_2342];\n                x_2354 = ((volatile __local\n                           int32_t *) scan_arr_mem_2348)[local_tid_2342];\n                if (sub32(local_tid_2342, mul32(squot32(local_tid_2342, 32),\n                                                32)) == 0) {\n                    x_2351 = x_2353;\n                    x_2352 = x_2354;\n                }\n            }\n        }\n        // in-block scan (hopefully no barriers needed)\n        {\n            skip_threads_2359 = 1;\n            while (slt32(skip_threads_2359, 32)) {\n                if (sle32(skip_threads_2359, sub32(local_tid_2342,\n                                                   mul32(squot32(local_tid_2342,\n                                                                 32), 32))) &&\n             ",
            "       (squot32(local_tid_2342, 32) == 0 && slt32(local_tid_2342,\n                                                               num_groups_1686))) {\n                    // read operands\n                    {\n                        x_2351 = ((volatile __local\n                                   int32_t *) scan_arr_mem_2346)[sub32(local_tid_2342,\n                                                                       skip_threads_2359)];\n                        x_2352 = ((volatile __local\n                                   int32_t *) scan_arr_mem_2348)[sub32(local_tid_2342,\n                                                                       skip_threads_2359)];\n                    }\n                    // perform operation\n                    {\n                        bool inactive_2360 =\n                             slt32(srem32(sub32(mul32(add32(sub32(add32(mul32(local_tid_2342,\n                                                                              32),\n                                                                        32), 1),\n                                                            1),\n                                                      mul32(segscan_group_sizze_1684,\n                                                            squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                      pts_per_node_at_lev_1013),\n                                                                                num_threads_2297),\n                                                                          1),\n                                                                    num_threads_2297))),\n                                                1), pts_per_node_at_lev_1013),\n                                   sub32(sub32(mul32(add32(sub32(add32(mul32(local_tid_2342,\n                                                                             32),\n                                                   ",
            "                    32), 1),\n                                                           1),\n                                                     mul32(segscan_group_sizze_1684,\n                                                           squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                     pts_per_node_at_lev_1013),\n                                                                               num_threads_2297),\n                                                                         1),\n                                                                   num_threads_2297))),\n                                               1),\n                                         sub32(mul32(add32(sub32(add32(mul32(sub32(local_tid_2342,\n                                                                                   skip_threads_2359),\n                                                                             32),\n                                                                       32), 1),\n                                                           1),\n                                                     mul32(segscan_group_sizze_1684,\n                                                           squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                     pts_per_node_at_lev_1013),\n                                                                               num_threads_2297),\n                                                                         1),\n                                                                   num_threads_2297))),\n                                               1)));\n                        \n                        if (inactive_2360) {\n                            x_2351 = x_2353;\n                            x_2352 = x_2354;\n                        }\n                        if (!inactive_2360) {\n             ",
            "               int32_t res_2355 = add32(x_2351, x_2353);\n                            int32_t res_2356 = add32(x_2352, x_2354);\n                            \n                            x_2351 = res_2355;\n                            x_2352 = res_2356;\n                        }\n                    }\n                }\n                if (sle32(wave_sizze_2344, skip_threads_2359)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                if (sle32(skip_threads_2359, sub32(local_tid_2342,\n                                                   mul32(squot32(local_tid_2342,\n                                                                 32), 32))) &&\n                    (squot32(local_tid_2342, 32) == 0 && slt32(local_tid_2342,\n                                                               num_groups_1686))) {\n                    // write result\n                    {\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_2346)[local_tid_2342] =\n                            x_2351;\n                        x_2353 = x_2351;\n                        ((volatile __local\n                          int32_t *) scan_arr_mem_2348)[local_tid_2342] =\n                            x_2352;\n                        x_2354 = x_2352;\n                    }\n                }\n                if (sle32(wave_sizze_2344, skip_threads_2359)) {\n                    barrier(CLK_LOCAL_MEM_FENCE);\n                }\n                skip_threads_2359 *= 2;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // carry-in for every block except the first\n    {\n        if (!(squot32(local_tid_2342, 32) == 0 || !slt32(local_tid_2342,\n                                                         num_groups_1686))) {\n            // read operands\n            {\n                x_2331 = x_2329;\n                x_2332 = x_2330;\n                x_2329 = ((__local\n                           int32_t *) scan_arr_mem_2346)[sub32(squot32(local_tid_2342,\n ",
            "                                                                      32), 1)];\n                x_2330 = ((__local\n                           int32_t *) scan_arr_mem_2348)[sub32(squot32(local_tid_2342,\n                                                                       32), 1)];\n            }\n            // perform operation\n            {\n                bool inactive_2361 =\n                     slt32(srem32(sub32(mul32(add32(local_tid_2342, 1),\n                                              mul32(segscan_group_sizze_1684,\n                                                    squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                              pts_per_node_at_lev_1013),\n                                                                        num_threads_2297),\n                                                                  1),\n                                                            num_threads_2297))),\n                                        1), pts_per_node_at_lev_1013),\n                           sub32(sub32(mul32(add32(local_tid_2342, 1),\n                                             mul32(segscan_group_sizze_1684,\n                                                   squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                             pts_per_node_at_lev_1013),\n                                                                       num_threads_2297),\n                                                                 1),\n                                                           num_threads_2297))),\n                                       1),\n                                 sub32(mul32(add32(sub32(mul32(squot32(local_tid_2342,\n                                                                       32), 32),\n                                                         1), 1),\n                                             mul32(segscan_group_sizze_1684,\n          ",
            "                                         squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                             pts_per_node_at_lev_1013),\n                                                                       num_threads_2297),\n                                                                 1),\n                                                           num_threads_2297))),\n                                       1)));\n                \n                if (inactive_2361) {\n                    x_2329 = x_2331;\n                    x_2330 = x_2332;\n                }\n                if (!inactive_2361) {\n                    int32_t res_2333 = add32(x_2329, x_2331);\n                    int32_t res_2334 = add32(x_2330, x_2332);\n                    \n                    x_2329 = res_2333;\n                    x_2330 = res_2334;\n                }\n            }\n            // write final result\n            {\n                ((__local int32_t *) scan_arr_mem_2346)[local_tid_2342] =\n                    x_2329;\n                ((__local int32_t *) scan_arr_mem_2348)[local_tid_2342] =\n                    x_2330;\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // restore correct values for first block\n    {\n        if (squot32(local_tid_2342, 32) == 0) {\n            ((__local int32_t *) scan_arr_mem_2346)[local_tid_2342] = x_2331;\n            ((__local int32_t *) scan_arr_mem_2348)[local_tid_2342] = x_2332;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // threads in bounds write scanned carries\n    {\n        if (slt32(gtid_1676, nodes_this_lvl_1009) && slt32(gtid_1688,\n                                                           pts_per_node_at_lev_1013)) {\n            ((__global int32_t *) mem_1977)[add32(mul32(gtid_1676,\n                                                        pts_per_node_at_lev_1013),\n                                                  gtid_1688)] = ((__local\n                            ",
            "                                      int32_t *) scan_arr_mem_2346)[local_tid_2342];\n            ((__global int32_t *) mem_1982)[add32(mul32(gtid_1676,\n                                                        pts_per_node_at_lev_1013),\n                                                  gtid_1688)] = ((__local\n                                                                  int32_t *) scan_arr_mem_2348)[local_tid_2342];\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_1684\n}\n__kernel void scan_stage3_1689(__global int *global_failure,\n                               int32_t nodes_this_lvl_1009,\n                               int32_t pts_per_node_at_lev_1013,\n                               int32_t num_groups_1686, __global\n                               unsigned char *mem_1977, __global\n                               unsigned char *mem_1982,\n                               int32_t num_threads_2297,\n                               int32_t required_groups_2362)\n{\n    #define segscan_group_sizze_1684 (mainzisegscan_group_sizze_1683)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2363;\n    int32_t local_tid_2364;\n    int32_t group_sizze_2367;\n    int32_t wave_sizze_2366;\n    int32_t group_tid_2365;\n    \n    global_tid_2363 = get_global_id(0);\n    local_tid_2364 = get_local_id(0);\n    group_sizze_2367 = get_local_size(0);\n    wave_sizze_2366 = LOCKSTEP_WIDTH;\n    group_tid_2365 = get_group_id(0);\n    \n    int32_t phys_tid_1689 = global_tid_2363;\n    int32_t phys_group_id_2368;\n    \n    phys_group_id_2368 = get_group_id(0);\n    for (int32_t i_2369 = 0; i_2369 <\n         squot32(sub32(add32(sub32(required_groups_2362, phys_group_id_2368),\n                             num_groups_1686), 1), num_groups_1686); i_2369++) {\n        int32_t virt_group_id_2370 = add32(phys_group_id_2368, mul32(i_2369,\n                                   ",
            "                                  num_groups_1686));\n        int32_t flat_idx_2371 = add32(mul32(virt_group_id_2370,\n                                            segscan_group_sizze_1684),\n                                      local_tid_2364);\n        int32_t gtid_1676 = squot32(flat_idx_2371, pts_per_node_at_lev_1013);\n        int32_t gtid_1688;\n        \n        gtid_1688 = sub32(flat_idx_2371, mul32(squot32(flat_idx_2371,\n                                                       pts_per_node_at_lev_1013),\n                                               pts_per_node_at_lev_1013));\n        \n        int32_t orig_group_2372 = squot32(flat_idx_2371,\n                                          mul32(segscan_group_sizze_1684,\n                                                squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                          pts_per_node_at_lev_1013),\n                                                                    num_threads_2297),\n                                                              1),\n                                                        num_threads_2297)));\n        int32_t carry_in_flat_idx_2373 = sub32(mul32(orig_group_2372,\n                                                     mul32(segscan_group_sizze_1684,\n                                                           squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                     pts_per_node_at_lev_1013),\n                                                                               num_threads_2297),\n                                                                         1),\n                                                                   num_threads_2297))),\n                                               1);\n        \n        if (slt32(gtid_1676, nodes_this_lvl_1009) && slt32(gtid_1688,\n                                                           pts_per_node_at_lev_1013)",
            ") {\n            if (!(orig_group_2372 == 0 || (flat_idx_2371 ==\n                                           sub32(mul32(add32(orig_group_2372,\n                                                             1),\n                                                       mul32(segscan_group_sizze_1684,\n                                                             squot32(sub32(add32(mul32(nodes_this_lvl_1009,\n                                                                                       pts_per_node_at_lev_1013),\n                                                                                 num_threads_2297),\n                                                                           1),\n                                                                     num_threads_2297))),\n                                                 1) ||\n                                           slt32(srem32(flat_idx_2371,\n                                                        pts_per_node_at_lev_1013),\n                                                 sub32(flat_idx_2371,\n                                                       carry_in_flat_idx_2373))))) {\n                int32_t x_2335;\n                int32_t x_2336;\n                int32_t x_2337;\n                int32_t x_2338;\n                \n                x_2335 = ((__global\n                           int32_t *) mem_1977)[add32(mul32(squot32(carry_in_flat_idx_2373,\n                                                                    pts_per_node_at_lev_1013),\n                                                            pts_per_node_at_lev_1013),\n                                                      sub32(carry_in_flat_idx_2373,\n                                                            mul32(squot32(carry_in_flat_idx_2373,\n                                                                          pts_per_node_at_lev_1013),\n                                                                  pts_per_node_at_lev_1013)))];\n                x",
            "_2336 = ((__global\n                           int32_t *) mem_1982)[add32(mul32(squot32(carry_in_flat_idx_2373,\n                                                                    pts_per_node_at_lev_1013),\n                                                            pts_per_node_at_lev_1013),\n                                                      sub32(carry_in_flat_idx_2373,\n                                                            mul32(squot32(carry_in_flat_idx_2373,\n                                                                          pts_per_node_at_lev_1013),\n                                                                  pts_per_node_at_lev_1013)))];\n                x_2337 = ((__global int32_t *) mem_1977)[add32(mul32(gtid_1676,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1688)];\n                x_2338 = ((__global int32_t *) mem_1982)[add32(mul32(gtid_1676,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1688)];\n                \n                int32_t res_2339 = add32(x_2335, x_2337);\n                int32_t res_2340 = add32(x_2336, x_2338);\n                \n                x_2335 = res_2339;\n                x_2336 = res_2340;\n                ((__global int32_t *) mem_1977)[add32(mul32(gtid_1676,\n                                                            pts_per_node_at_lev_1013),\n                                                      gtid_1688)] = x_2335;\n                ((__global int32_t *) mem_1982)[add32(mul32(gtid_1676,\n                                                            pts_per_node_at_lev_1013),\n                                                      gtid_1688)] = x_2336;\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segscan_group_sizze_1684\n}\n_",
            "_kernel void segmap_1491(__global int *global_failure,\n                          int failure_is_an_option, __global\n                          int *global_failure_args, int32_t d_912,\n                          int32_t res_942, int32_t conc_tmp_972,\n                          int32_t lev_1008, int32_t nodes_this_lvl_1009,\n                          int32_t segmap_usable_groups_1501, __global\n                          unsigned char *mem_1920, __global\n                          unsigned char *mem_1926, __global\n                          unsigned char *mem_1929, __global\n                          unsigned char *mem_1940, __global\n                          unsigned char *mem_1943, __global\n                          unsigned char *mem_1946)\n{\n    #define segmap_group_sizze_1495 (mainzisegmap_group_sizze_1494)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2253;\n    int32_t local_tid_2254;\n    int32_t group_sizze_2257;\n    int32_t wave_sizze_2256;\n    int32_t group_tid_2255;\n    \n    global_tid_2253 = get_global_id(0);\n    local_tid_2254 = get_local_id(0);\n    group_sizze_2257 = get_local_size(0);\n    wave_sizze_2256 = LOCKSTEP_WIDTH;\n    group_tid_2255 = get_group_id(0);\n    \n    int32_t phys_tid_1491 = global_tid_2253;\n    int32_t gtid_1490 = add32(mul32(group_tid_2255, segmap_group_sizze_1495),\n                              local_tid_2254);\n    \n    if (slt32(gtid_1490, nodes_this_lvl_1009)) {\n        int32_t x_1503 = add32(nodes_this_lvl_1009, gtid_1490);\n        int32_t node_ind_1504 = sub32(x_1503, 1);\n        \n        for (int32_t i_2258 = 0; i_2258 < conc_tmp_972; i_2258++) {\n            ((__global float *) mem_1940)[add32(phys_tid_1491, mul32(i_2258,\n                                                                     mul32(segmap_usable_groups_1501,\n                                                                           segmap_group_sizze",
            "_1495)))] =\n                ((__global float *) mem_1920)[i_2258];\n        }\n        \n        int32_t x_1506 = add32(1, node_ind_1504);\n        int32_t res_1507;\n        int32_t ancestor_1509 = 0;\n        \n        for (int32_t i_1511 = 0; i_1511 < lev_1008; i_1511++) {\n            int32_t x_1512 = sub32(lev_1008, i_1511);\n            int32_t k_1513 = sub32(x_1512, 1);\n            int32_t tpk_1514 = 1 << k_1513;\n            int32_t x_1515 = sub32(x_1506, tpk_1514);\n            bool zzero_1516 = tpk_1514 == 0;\n            bool nonzzero_1517 = !zzero_1516;\n            bool nonzzero_cert_1518;\n            \n            if (!nonzzero_1517) {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 0) == -1) {\n                    ;\n                }\n                return;\n            }\n            \n            int32_t res_1519 = sdiv32(x_1515, tpk_1514);\n            bool x_1520 = sle32(0, ancestor_1509);\n            bool y_1521 = slt32(ancestor_1509, res_942);\n            bool bounds_check_1522 = x_1520 && y_1521;\n            bool index_certs_1523;\n            \n            if (!bounds_check_1522) {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 1) == -1) {\n                    global_failure_args[0] = ancestor_1509;\n                    global_failure_args[1] = res_942;\n                    ;\n                }\n                return;\n            }\n            \n            int32_t anc_dim_1524 = ((__global\n                                     int32_t *) mem_1929)[ancestor_1509];\n            int32_t x_1525 = 1 & res_1519;\n            bool cond_1526 = x_1525 == 0;\n            int32_t lub_ind_1527;\n            \n            if (cond_1526) {\n                lub_ind_1527 = anc_dim_1524;\n            } else {\n                int32_t res_1528 = add32(d_912, anc_dim_1524);\n                \n                lub_ind_1527 = res_1528;\n            }\n            \n            float anc_med_1529 = ((__global float *) mem_1926)[ancestor_1509];\n            bool res_153",
            "0;\n            \n            res_1530 = futrts_isinf32(anc_med_1529);\n            \n            bool cond_1531 = !res_1530;\n            float lw_val_1532;\n            \n            if (cond_1531) {\n                lw_val_1532 = anc_med_1529;\n            } else {\n                bool x_1533 = sle32(0, lub_ind_1527);\n                bool y_1534 = slt32(lub_ind_1527, conc_tmp_972);\n                bool bounds_check_1535 = x_1533 && y_1534;\n                bool index_certs_1536;\n                \n                if (!bounds_check_1535) {\n                    if (atomic_cmpxchg_i32_global(global_failure, -1, 2) ==\n                        -1) {\n                        global_failure_args[0] = lub_ind_1527;\n                        global_failure_args[1] = conc_tmp_972;\n                        ;\n                    }\n                    return;\n                }\n                \n                float res_1537 = ((__global\n                                   float *) mem_1940)[add32(phys_tid_1491,\n                                                            mul32(lub_ind_1527,\n                                                                  mul32(segmap_usable_groups_1501,\n                                                                        segmap_group_sizze_1495)))];\n                \n                lw_val_1532 = res_1537;\n            }\n            \n            bool x_1538 = sle32(0, lub_ind_1527);\n            bool y_1539 = slt32(lub_ind_1527, conc_tmp_972);\n            bool bounds_check_1540 = x_1538 && y_1539;\n            bool index_certs_1541;\n            \n            if (!bounds_check_1540) {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 3) == -1) {\n                    global_failure_args[0] = lub_ind_1527;\n                    global_failure_args[1] = conc_tmp_972;\n                    ;\n                }\n                return;\n            }\n            ((__global float *) mem_1940)[add32(phys_tid_1491,\n                                                m",
            "ul32(lub_ind_1527,\n                                                      mul32(segmap_usable_groups_1501,\n                                                            segmap_group_sizze_1495)))] =\n                lw_val_1532;\n            \n            int32_t ancestor_tmp_2259 = res_1519;\n            \n            ancestor_1509 = ancestor_tmp_2259;\n        }\n        res_1507 = ancestor_1509;\n        \n        int32_t res_1544;\n        float res_1545;\n        int32_t redout_1883;\n        float redout_1884;\n        \n        redout_1883 = -1;\n        redout_1884 = -INFINITY;\n        for (int32_t i_1885 = 0; i_1885 < d_912; i_1885++) {\n            float res_elem_1554 = ((__global\n                                    float *) mem_1940)[add32(phys_tid_1491,\n                                                             mul32(i_1885,\n                                                                   mul32(segmap_usable_groups_1501,\n                                                                         segmap_group_sizze_1495)))];\n            int32_t i_1555 = add32(d_912, i_1885);\n            float x_1556 = ((__global float *) mem_1940)[add32(phys_tid_1491,\n                                                               mul32(i_1555,\n                                                                     mul32(segmap_usable_groups_1501,\n                                                                           segmap_group_sizze_1495)))];\n            float abs_arg_1557 = x_1556 - res_elem_1554;\n            float res_1558 = (float) fabs(abs_arg_1557);\n            bool cond_1550 = res_1558 <= redout_1884;\n            int32_t res_1551;\n            \n            if (cond_1550) {\n                res_1551 = redout_1883;\n            } else {\n                res_1551 = i_1885;\n            }\n            \n            float res_1552;\n            \n            if (cond_1550) {\n                res_1552 = redout_1884;\n            } else {\n                res_1552 = res_1558;\n            }\n         ",
            "   \n            int32_t redout_tmp_2261 = res_1551;\n            float redout_tmp_2262;\n            \n            redout_tmp_2262 = res_1552;\n            redout_1883 = redout_tmp_2261;\n            redout_1884 = redout_tmp_2262;\n        }\n        res_1544 = redout_1883;\n        res_1545 = redout_1884;\n        \n        bool cond_1559 = node_ind_1504 == 0;\n        bool cond_1560 = !cond_1559;\n        bool res_1561;\n        int32_t res_1562;\n        int32_t res_1563;\n        bool loop_while_1564;\n        int32_t cur_node_1565;\n        int32_t res_ind_1566;\n        \n        loop_while_1564 = cond_1560;\n        cur_node_1565 = node_ind_1504;\n        res_ind_1566 = -1;\n        while (loop_while_1564) {\n            int32_t x_1567 = sub32(cur_node_1565, 1);\n            int32_t res_1568 = sdiv32(x_1567, 2);\n            bool x_1569 = sle32(0, res_1568);\n            bool y_1570 = slt32(res_1568, res_942);\n            bool bounds_check_1571 = x_1569 && y_1570;\n            bool index_certs_1572;\n            \n            if (!bounds_check_1571) {\n                if (atomic_cmpxchg_i32_global(global_failure, -1, 4) == -1) {\n                    global_failure_args[0] = res_1568;\n                    global_failure_args[1] = res_942;\n                    ;\n                }\n                return;\n            }\n            \n            int32_t x_1573 = ((__global int32_t *) mem_1929)[res_1568];\n            bool cond_1574 = x_1573 == res_1544;\n            int32_t res_ind_1575;\n            \n            if (cond_1574) {\n                res_ind_1575 = res_1568;\n            } else {\n                res_ind_1575 = -1;\n            }\n            \n            bool cond_1576 = res_1568 == 0;\n            bool cond_1577 = !cond_1576;\n            bool eq_x_y_1578 = -1 == res_1568;\n            bool p_and_eq_x_y_1579 = cond_1574 && eq_x_y_1578;\n            bool not_p_1580 = !cond_1574;\n            bool res_1581 = p_and_eq_x_y_1579 || not_p_1580;\n            bool x_1582 = cond_1577 && res_1581;\n        ",
            "    bool loop_while_tmp_2263 = x_1582;\n            int32_t cur_node_tmp_2264 = res_1568;\n            int32_t res_ind_tmp_2265;\n            \n            res_ind_tmp_2265 = res_ind_1575;\n            loop_while_1564 = loop_while_tmp_2263;\n            cur_node_1565 = cur_node_tmp_2264;\n            res_ind_1566 = res_ind_tmp_2265;\n        }\n        res_1561 = loop_while_1564;\n        res_1562 = cur_node_1565;\n        res_1563 = res_ind_1566;\n        ((__global int32_t *) mem_1943)[gtid_1490] = res_1544;\n        ((__global int32_t *) mem_1946)[gtid_1490] = res_1563;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1495\n}\n__kernel void segmap_1587(__global int *global_failure,\n                          int failure_is_an_option, __global\n                          int *global_failure_args, int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013, __global\n                          unsigned char *indir_mem_1933, __global\n                          unsigned char *mem_1958, __global\n                          unsigned char *res_r_mem_2032, __global\n                          unsigned char *mem_2038, __global\n                          unsigned char *mem_2043)\n{\n    #define segmap_group_sizze_1593 (mainzisegmap_group_sizze_1592)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2394;\n    int32_t local_tid_2395;\n    int32_t group_sizze_2398;\n    int32_t wave_sizze_2397;\n    int32_t group_tid_2396;\n    \n    global_tid_2394 = get_global_id(0);\n    local_tid_2395 = get_local_id(0);\n    group_sizze_2398 = get_local_size(0);\n    wave_sizze_2397 = LOCKSTEP_WIDTH;\n    group_tid_2396 = get_group_id(0);\n    \n    int32_t phys_tid_1587 = global_tid_2394;\n    int32_t gtid_1585 = squot32(add32(mul32(group_tid_2396,\n                                            segmap_group_sizze_1593),\n                                      local_ti",
            "d_2395),\n                                pts_per_node_at_lev_1013);\n    int32_t gtid_1586;\n    \n    gtid_1586 = sub32(add32(mul32(group_tid_2396, segmap_group_sizze_1593),\n                            local_tid_2395),\n                      mul32(squot32(add32(mul32(group_tid_2396,\n                                                segmap_group_sizze_1593),\n                                          local_tid_2395),\n                                    pts_per_node_at_lev_1013),\n                            pts_per_node_at_lev_1013));\n    if (slt32(gtid_1585, nodes_this_lvl_1009) && slt32(gtid_1586,\n                                                       pts_per_node_at_lev_1013)) {\n        int32_t x_1602 = ((__global\n                           int32_t *) res_r_mem_2032)[add32(mul32(gtid_1585,\n                                                                  pts_per_node_at_lev_1013),\n                                                            gtid_1586)];\n        float res_1603 = ((__global float *) mem_1958)[add32(mul32(gtid_1585,\n                                                                   pts_per_node_at_lev_1013),\n                                                             x_1602)];\n        bool x_1604 = sle32(0, x_1602);\n        bool y_1605 = slt32(x_1602, pts_per_node_at_lev_1013);\n        bool bounds_check_1606 = x_1604 && y_1605;\n        bool index_certs_1607;\n        \n        if (!bounds_check_1606) {\n            if (atomic_cmpxchg_i32_global(global_failure, -1, 6) == -1) {\n                global_failure_args[0] = x_1602;\n                global_failure_args[1] = pts_per_node_at_lev_1013;\n                ;\n            }\n            return;\n        }\n        \n        int32_t binop_x_1863 = mul32(pts_per_node_at_lev_1013, gtid_1585);\n        int32_t new_index_1864 = add32(x_1602, binop_x_1863);\n        int32_t res_1608 = ((__global\n                             int32_t *) indir_mem_1933)[new_index_1864];\n        \n        ((__global int32_t *) mem_2038)[add32(mul",
            "32(gtid_1585,\n                                                    pts_per_node_at_lev_1013),\n                                              gtid_1586)] = res_1608;\n        ((__global float *) mem_2043)[add32(mul32(gtid_1585,\n                                                  pts_per_node_at_lev_1013),\n                                            gtid_1586)] = res_1603;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1593\n}\n__kernel void segmap_1622(__global int *global_failure,\n                          int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013, __global\n                          unsigned char *xs_expanded_mem_1964, __global\n                          unsigned char *xs_expanded_mem_1965, __global\n                          unsigned char *mem_1971, __global\n                          unsigned char *mem_1977, __global\n                          unsigned char *mem_1982, __global\n                          unsigned char *mem_1987, __global\n                          unsigned char *mem_2009, __global\n                          unsigned char *mem_2013, __global\n                          unsigned char *mem_2017)\n{\n    #define segmap_group_sizze_1628 (mainzisegmap_group_sizze_1627)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2389;\n    int32_t local_tid_2390;\n    int32_t group_sizze_2393;\n    int32_t wave_sizze_2392;\n    int32_t group_tid_2391;\n    \n    global_tid_2389 = get_global_id(0);\n    local_tid_2390 = get_local_id(0);\n    group_sizze_2393 = get_local_size(0);\n    wave_sizze_2392 = LOCKSTEP_WIDTH;\n    group_tid_2391 = get_group_id(0);\n    \n    int32_t phys_tid_1622 = global_tid_2389;\n    int32_t gtid_1620 = squot32(add32(mul32(group_tid_2391,\n                                            segmap_group_sizze_1628),\n                                      local_tid_2390),\n                        ",
            "        pts_per_node_at_lev_1013);\n    int32_t gtid_1621;\n    \n    gtid_1621 = sub32(add32(mul32(group_tid_2391, segmap_group_sizze_1628),\n                            local_tid_2390),\n                      mul32(squot32(add32(mul32(group_tid_2391,\n                                                segmap_group_sizze_1628),\n                                          local_tid_2390),\n                                    pts_per_node_at_lev_1013),\n                            pts_per_node_at_lev_1013));\n    if (slt32(gtid_1620, nodes_this_lvl_1009) && slt32(gtid_1621,\n                                                       pts_per_node_at_lev_1013)) {\n        int32_t res_1641 = ((__global int32_t *) mem_2009)[gtid_1620];\n        int32_t x_1644 = ((__global int32_t *) mem_1987)[add32(mul32(gtid_1620,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1621)];\n        int32_t x_1645 = ((__global int32_t *) mem_1977)[add32(mul32(gtid_1620,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1621)];\n        int32_t x_1646 = ((__global int32_t *) mem_1982)[add32(mul32(gtid_1620,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1621)];\n        int32_t x_1647 = ((__global int32_t *) mem_1971)[add32(mul32(gtid_1620,\n                                                                     pts_per_node_at_lev_1013),\n                                                               gtid_1621)];\n        float write_value_1648 = ((__global\n                                   float *) xs_expanded_mem_1964)[add32(mul32(gtid_1620,\n                                                                              pts_per_node_at_lev_1013),\n                                 ",
            "                                       gtid_1621)];\n        int32_t write_value_1649 = ((__global\n                                     int32_t *) xs_expanded_mem_1965)[add32(mul32(gtid_1620,\n                                                                                  pts_per_node_at_lev_1013),\n                                                                            gtid_1621)];\n        int32_t res_1650 = mul32(x_1644, x_1645);\n        int32_t res_1651 = add32(res_1641, x_1646);\n        int32_t res_1652 = mul32(x_1647, res_1651);\n        int32_t res_1653 = add32(res_1650, res_1652);\n        int32_t res_1654 = sub32(res_1653, 1);\n        \n        if ((sle32(0, gtid_1620) && slt32(gtid_1620, nodes_this_lvl_1009)) &&\n            (sle32(0, res_1654) && slt32(res_1654, pts_per_node_at_lev_1013))) {\n            ((__global float *) mem_2017)[add32(mul32(res_1654,\n                                                      nodes_this_lvl_1009),\n                                                gtid_1620)] = write_value_1648;\n        }\n        if ((sle32(0, gtid_1620) && slt32(gtid_1620, nodes_this_lvl_1009)) &&\n            (sle32(0, res_1654) && slt32(res_1654, pts_per_node_at_lev_1013))) {\n            ((__global int32_t *) mem_2013)[add32(mul32(res_1654,\n                                                        nodes_this_lvl_1009),\n                                                  gtid_1620)] =\n                write_value_1649;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1628\n}\n__kernel void segmap_1656(__global int *global_failure,\n                          int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013,\n                          int32_t num_groups_1662, __global\n                          unsigned char *mem_1991, __global\n                          unsigned char *mem_1995, __global\n                          unsigned char *mem_1999, __global\n                          unsigned char *mem_2003, __global\n     ",
            "                     unsigned char *mem_2006, __global\n                          unsigned char *mem_2009, __global\n                          unsigned char *mem_2013, __global\n                          unsigned char *mem_2017)\n{\n    #define segmap_group_sizze_1660 (mainzisegmap_group_sizze_1659)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2374;\n    int32_t local_tid_2375;\n    int32_t group_sizze_2378;\n    int32_t wave_sizze_2377;\n    int32_t group_tid_2376;\n    \n    global_tid_2374 = get_global_id(0);\n    local_tid_2375 = get_local_id(0);\n    group_sizze_2378 = get_local_size(0);\n    wave_sizze_2377 = LOCKSTEP_WIDTH;\n    group_tid_2376 = get_group_id(0);\n    \n    int32_t phys_tid_1656 = global_tid_2374;\n    int32_t phys_group_id_2379;\n    \n    phys_group_id_2379 = get_group_id(0);\n    for (int32_t i_2380 = 0; i_2380 <\n         squot32(sub32(add32(sub32(squot32(sub32(add32(nodes_this_lvl_1009,\n                                                       segmap_group_sizze_1660),\n                                                 1), segmap_group_sizze_1660),\n                                   phys_group_id_2379), num_groups_1662), 1),\n                 num_groups_1662); i_2380++) {\n        int32_t virt_group_id_2381 = add32(phys_group_id_2379, mul32(i_2380,\n                                                                     num_groups_1662));\n        int32_t gtid_1655 = add32(mul32(virt_group_id_2381,\n                                        segmap_group_sizze_1660),\n                                  local_tid_2375);\n        \n        if (slt32(gtid_1655, nodes_this_lvl_1009)) {\n            int32_t res_1667;\n            int32_t redout_1888 = 0;\n            \n            for (int32_t i_1891 = 0; i_1891 < pts_per_node_at_lev_1013;\n                 i_1891++) {\n                int32_t x_1674 = ((__global\n                                   int32_t *) mem_1",
            "999)[add32(mul32(i_1891,\n                                                                    nodes_this_lvl_1009),\n                                                              gtid_1655)];\n                int32_t res_1672 = add32(x_1674, redout_1888);\n                \n                for (int32_t i_2385 = 0; i_2385 < 1; i_2385++) {\n                    ((__global int32_t *) mem_2003)[add32(phys_tid_1656,\n                                                          mul32(add32(i_1891,\n                                                                      i_2385),\n                                                                mul32(num_groups_1662,\n                                                                      segmap_group_sizze_1660)))] =\n                        ((__global\n                          int32_t *) mem_1995)[add32(add32(mul32(nodes_this_lvl_1009,\n                                                                 i_1891),\n                                                           gtid_1655),\n                                                     mul32(i_2385,\n                                                           nodes_this_lvl_1009))];\n                }\n                for (int32_t i_2386 = 0; i_2386 < 1; i_2386++) {\n                    ((__global float *) mem_2006)[add32(phys_tid_1656,\n                                                        mul32(add32(i_1891,\n                                                                    i_2386),\n                                                              mul32(num_groups_1662,\n                                                                    segmap_group_sizze_1660)))] =\n                        ((__global\n                          float *) mem_1991)[add32(add32(mul32(nodes_this_lvl_1009,\n                                                               i_1891),\n                                                         gtid_1655),\n                                                   mul32(i_2386,\n                 ",
            "                                        nodes_this_lvl_1009))];\n                }\n                \n                int32_t redout_tmp_2382 = res_1672;\n                \n                redout_1888 = redout_tmp_2382;\n            }\n            res_1667 = redout_1888;\n            ((__global int32_t *) mem_2009)[gtid_1655] = res_1667;\n            for (int32_t i_2387 = 0; i_2387 < pts_per_node_at_lev_1013;\n                 i_2387++) {\n                ((__global int32_t *) mem_2013)[add32(mul32(i_2387,\n                                                            nodes_this_lvl_1009),\n                                                      gtid_1655)] = ((__global\n                                                                      int32_t *) mem_2003)[add32(phys_tid_1656,\n                                                                                                 mul32(i_2387,\n                                                                                                       mul32(num_groups_1662,\n                                                                                                             segmap_group_sizze_1660)))];\n            }\n            for (int32_t i_2388 = 0; i_2388 < pts_per_node_at_lev_1013;\n                 i_2388++) {\n                ((__global float *) mem_2017)[add32(mul32(i_2388,\n                                                          nodes_this_lvl_1009),\n                                                    gtid_1655)] = ((__global\n                                                                    float *) mem_2006)[add32(phys_tid_1656,\n                                                                                             mul32(i_2388,\n                                                                                                   mul32(num_groups_1662,\n                                                                                                         segmap_group_sizze_1660)))];\n            }\n        }\n        barrier(CL",
            "K_GLOBAL_MEM_FENCE);\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1660\n}\n__kernel void segmap_1702(__global int *global_failure,\n                          int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013, int32_t i_1241,\n                          unsigned char res_1740, __global\n                          unsigned char *xs_expanded_mem_1964, __global\n                          unsigned char *mem_1971)\n{\n    #define segmap_group_sizze_1708 (mainzisegmap_group_sizze_1707)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2292;\n    int32_t local_tid_2293;\n    int32_t group_sizze_2296;\n    int32_t wave_sizze_2295;\n    int32_t group_tid_2294;\n    \n    global_tid_2292 = get_global_id(0);\n    local_tid_2293 = get_local_id(0);\n    group_sizze_2296 = get_local_size(0);\n    wave_sizze_2295 = LOCKSTEP_WIDTH;\n    group_tid_2294 = get_group_id(0);\n    \n    int32_t phys_tid_1702 = global_tid_2292;\n    int32_t gtid_1700 = squot32(add32(mul32(group_tid_2294,\n                                            segmap_group_sizze_1708),\n                                      local_tid_2293),\n                                pts_per_node_at_lev_1013);\n    int32_t gtid_1701;\n    \n    gtid_1701 = sub32(add32(mul32(group_tid_2294, segmap_group_sizze_1708),\n                            local_tid_2293),\n                      mul32(squot32(add32(mul32(group_tid_2294,\n                                                segmap_group_sizze_1708),\n                                          local_tid_2293),\n                                    pts_per_node_at_lev_1013),\n                            pts_per_node_at_lev_1013));\n    if (slt32(gtid_1700, nodes_this_lvl_1009) && slt32(gtid_1701,\n                                                       pts_per_node_at_lev_1013)) {\n        float x_1716 = ((__global\n                         float",
            " *) xs_expanded_mem_1964)[add32(mul32(gtid_1700,\n                                                                    pts_per_node_at_lev_1013),\n                                                              gtid_1701)];\n        int32_t i32_arg_1717;\n        \n        i32_arg_1717 = futrts_to_bits32(x_1716);\n        \n        int32_t unsign_arg_1718 = ashr32(i32_arg_1717, i_1241);\n        int32_t unsign_arg_1719 = 1 & unsign_arg_1718;\n        int32_t unsign_arg_1720 = ashr32(i32_arg_1717, 31);\n        int32_t unsign_arg_1721 = 1 & unsign_arg_1720;\n        bool cond_1722 = unsign_arg_1721 == 1;\n        bool x_1723 = !cond_1722;\n        bool y_1724 = x_1723 && res_1740;\n        bool cond_1725 = cond_1722 || y_1724;\n        int32_t res_1726;\n        \n        if (cond_1725) {\n            int32_t res_1727 = 1 ^ unsign_arg_1719;\n            \n            res_1726 = res_1727;\n        } else {\n            res_1726 = unsign_arg_1719;\n        }\n        ((__global int32_t *) mem_1971)[add32(mul32(gtid_1700,\n                                                    pts_per_node_at_lev_1013),\n                                              gtid_1701)] = res_1726;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1708\n}\n__kernel void segmap_1744(__global int *global_failure,\n                          int failure_is_an_option, __global\n                          int *global_failure_args, int32_t m_911,\n                          int32_t d_912, int32_t res_943,\n                          int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013, __global\n                          unsigned char *input_mem_1905, __global\n                          unsigned char *indir_mem_1933, __global\n                          unsigned char *mem_1943, __global\n                          unsigned char *mem_1952, __global\n                          unsigned char *mem_1958)\n{\n    #define segmap_group_sizze_1750 (mainzisegmap_group_sizze_1749)\n    \n    const int block_dim0 = 0;",
            "\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2276;\n    int32_t local_tid_2277;\n    int32_t group_sizze_2280;\n    int32_t wave_sizze_2279;\n    int32_t group_tid_2278;\n    \n    global_tid_2276 = get_global_id(0);\n    local_tid_2277 = get_local_id(0);\n    group_sizze_2280 = get_local_size(0);\n    wave_sizze_2279 = LOCKSTEP_WIDTH;\n    group_tid_2278 = get_group_id(0);\n    \n    int32_t phys_tid_1744 = global_tid_2276;\n    int32_t gtid_1742 = squot32(add32(mul32(group_tid_2278,\n                                            segmap_group_sizze_1750),\n                                      local_tid_2277),\n                                pts_per_node_at_lev_1013);\n    int32_t gtid_1743;\n    \n    gtid_1743 = sub32(add32(mul32(group_tid_2278, segmap_group_sizze_1750),\n                            local_tid_2277),\n                      mul32(squot32(add32(mul32(group_tid_2278,\n                                                segmap_group_sizze_1750),\n                                          local_tid_2277),\n                                    pts_per_node_at_lev_1013),\n                            pts_per_node_at_lev_1013));\n    if (slt32(gtid_1742, nodes_this_lvl_1009) && slt32(gtid_1743,\n                                                       pts_per_node_at_lev_1013)) {\n        int32_t x_1757 = ((__global int32_t *) mem_1943)[gtid_1742];\n        bool bounds_check_1758 = ((__global bool *) mem_1952)[gtid_1742];\n        int32_t binop_x_1844 = mul32(pts_per_node_at_lev_1013, gtid_1742);\n        int32_t new_index_1845 = add32(gtid_1743, binop_x_1844);\n        int32_t x_1759 = ((__global int32_t *) indir_mem_1933)[new_index_1845];\n        bool x_1760 = sle32(0, x_1759);\n        bool y_1761 = slt32(x_1759, res_943);\n        bool bounds_check_1762 = x_1760 && y_1761;\n        bool index_ok_1763 = bounds_check_1758 && bounds_check_1762;\n        bool index_certs_1764;\n        \n        if (!index_o",
            "k_1763) {\n            if (atomic_cmpxchg_i32_global(global_failure, -1, 5) == -1) {\n                global_failure_args[0] = x_1759;\n                global_failure_args[1] = x_1757;\n                global_failure_args[2] = res_943;\n                global_failure_args[3] = d_912;\n                ;\n            }\n            return;\n        }\n        \n        bool index_concat_cmp_1765 = sle32(m_911, x_1759);\n        float index_concat_branch_1766;\n        \n        if (index_concat_cmp_1765) {\n            index_concat_branch_1766 = INFINITY;\n        } else {\n            float index_concat_1767 = ((__global\n                                        float *) input_mem_1905)[add32(mul32(x_1759,\n                                                                             d_912),\n                                                                       x_1757)];\n            \n            index_concat_branch_1766 = index_concat_1767;\n        }\n        \n        float res_1768 = index_concat_branch_1766;\n        \n        ((__global float *) mem_1958)[add32(mul32(gtid_1742,\n                                                  pts_per_node_at_lev_1013),\n                                            gtid_1743)] = res_1768;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1750\n}\n__kernel void segmap_1770(__global int *global_failure, int32_t d_912,\n                          int32_t nodes_this_lvl_1009, __global\n                          unsigned char *mem_1943, __global\n                          unsigned char *mem_1952)\n{\n    #define segmap_group_sizze_1774 (mainzisegmap_group_sizze_1773)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2271;\n    int32_t local_tid_2272;\n    int32_t group_sizze_2275;\n    int32_t wave_sizze_2274;\n    int32_t group_tid_2273;\n    \n    global_tid_2271 = get_global_id(0);\n    local_tid_2272 = get_local_id(0);\n    group_sizze_2",
            "275 = get_local_size(0);\n    wave_sizze_2274 = LOCKSTEP_WIDTH;\n    group_tid_2273 = get_group_id(0);\n    \n    int32_t phys_tid_1770 = global_tid_2271;\n    int32_t gtid_1769 = add32(mul32(group_tid_2273, segmap_group_sizze_1774),\n                              local_tid_2272);\n    \n    if (slt32(gtid_1769, nodes_this_lvl_1009)) {\n        int32_t x_1781 = ((__global int32_t *) mem_1943)[gtid_1769];\n        bool x_1782 = sle32(0, x_1781);\n        bool y_1783 = slt32(x_1781, d_912);\n        bool bounds_check_1784 = x_1782 && y_1783;\n        \n        ((__global bool *) mem_1952)[gtid_1769] = bounds_check_1784;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1774\n}\n__kernel void segmap_1786(__global int *global_failure, int32_t res_942,\n                          int32_t nodes_this_lvl_1009,\n                          int32_t pts_per_node_at_lev_1013, int32_t mi_1208,\n                          int32_t i_1213, int32_t y_1324, __global\n                          unsigned char *mem_1926, __global\n                          unsigned char *mem_1929, __global\n                          unsigned char *mem_1932, __global\n                          unsigned char *mem_1943, __global\n                          unsigned char *mem_1946, __global\n                          unsigned char *mem_2043)\n{\n    #define segmap_group_sizze_1790 (mainzisegmap_group_sizze_1789)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2399;\n    int32_t local_tid_2400;\n    int32_t group_sizze_2403;\n    int32_t wave_sizze_2402;\n    int32_t group_tid_2401;\n    \n    global_tid_2399 = get_global_id(0);\n    local_tid_2400 = get_local_id(0);\n    group_sizze_2403 = get_local_size(0);\n    wave_sizze_2402 = LOCKSTEP_WIDTH;\n    group_tid_2401 = get_group_id(0);\n    \n    int32_t phys_tid_1786 = global_tid_2399;\n    int32_t write_i_1785 = add32(mul32(group_tid_2401, segmap_group_sizze_1790",
            "),\n                                 local_tid_2400);\n    \n    if (slt32(write_i_1785, nodes_this_lvl_1009)) {\n        int32_t write_value_1328 = ((__global\n                                     int32_t *) mem_1943)[write_i_1785];\n        int32_t write_value_1329 = ((__global\n                                     int32_t *) mem_1946)[write_i_1785];\n        float x_1332 = ((__global float *) mem_2043)[add32(mul32(write_i_1785,\n                                                                 pts_per_node_at_lev_1013),\n                                                           mi_1208)];\n        float y_1333 = ((__global float *) mem_2043)[add32(mul32(write_i_1785,\n                                                                 pts_per_node_at_lev_1013),\n                                                           i_1213)];\n        float x_1334 = x_1332 + y_1333;\n        float res_1335 = x_1334 / 2.0F;\n        int32_t res_1336 = add32(y_1324, write_i_1785);\n        \n        if (sle32(0, res_1336) && slt32(res_1336, res_942)) {\n            ((__global int32_t *) mem_1932)[res_1336] = write_value_1329;\n        }\n        if (sle32(0, res_1336) && slt32(res_1336, res_942)) {\n            ((__global float *) mem_1926)[res_1336] = res_1335;\n        }\n        if (sle32(0, res_1336) && slt32(res_1336, res_942)) {\n            ((__global int32_t *) mem_1929)[res_1336] = write_value_1328;\n        }\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1790\n}\n__kernel void segmap_1801(__global int *global_failure,\n                          int failure_is_an_option, __global\n                          int *global_failure_args, int32_t d_912,\n                          int32_t res_943, __global\n                          unsigned char *input_mem_1905, __global\n                          unsigned char *res_mem_2045, __global\n                          unsigned char *mem_2051, __global\n                          unsigned char *mem_2053, __global\n                          unsigned char *",
            "mem_2059)\n{\n    #define segmap_group_sizze_1807 (mainzisegmap_group_sizze_1806)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2409;\n    int32_t local_tid_2410;\n    int32_t group_sizze_2413;\n    int32_t wave_sizze_2412;\n    int32_t group_tid_2411;\n    \n    global_tid_2409 = get_global_id(0);\n    local_tid_2410 = get_local_id(0);\n    group_sizze_2413 = get_local_size(0);\n    wave_sizze_2412 = LOCKSTEP_WIDTH;\n    group_tid_2411 = get_group_id(0);\n    \n    int32_t phys_tid_1801 = global_tid_2409;\n    int32_t gtid_1799 = squot32(add32(mul32(group_tid_2411,\n                                            segmap_group_sizze_1807),\n                                      local_tid_2410), d_912);\n    int32_t gtid_1800;\n    \n    gtid_1800 = sub32(add32(mul32(group_tid_2411, segmap_group_sizze_1807),\n                            local_tid_2410),\n                      mul32(squot32(add32(mul32(group_tid_2411,\n                                                segmap_group_sizze_1807),\n                                          local_tid_2410), d_912), d_912));\n    if (slt32(gtid_1799, res_943) && slt32(gtid_1800, d_912)) {\n        int32_t x_1814 = ((__global int32_t *) res_mem_2045)[gtid_1799];\n        bool bounds_check_1815 = ((__global bool *) mem_2051)[gtid_1799];\n        bool index_concat_cmp_1816 = ((__global bool *) mem_2053)[gtid_1799];\n        bool index_certs_1818;\n        \n        if (!bounds_check_1815) {\n            if (atomic_cmpxchg_i32_global(global_failure, -1, 7) == -1) {\n                global_failure_args[0] = x_1814;\n                global_failure_args[1] = gtid_1800;\n                global_failure_args[2] = res_943;\n                global_failure_args[3] = d_912;\n                ;\n            }\n            return;\n        }\n        \n        float index_concat_branch_1819;\n        \n        if (index_concat_cmp_1816) {\n            index_concat_",
            "branch_1819 = INFINITY;\n        } else {\n            float index_concat_1820 = ((__global\n                                        float *) input_mem_1905)[add32(mul32(x_1814,\n                                                                             d_912),\n                                                                       gtid_1800)];\n            \n            index_concat_branch_1819 = index_concat_1820;\n        }\n        \n        float res_1821 = index_concat_branch_1819;\n        \n        ((__global float *) mem_2059)[add32(mul32(gtid_1799, d_912),\n                                            gtid_1800)] = res_1821;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1807\n}\n__kernel void segmap_1823(__global int *global_failure, int32_t m_911,\n                          int32_t res_943, __global unsigned char *res_mem_2045,\n                          __global unsigned char *mem_2051, __global\n                          unsigned char *mem_2053)\n{\n    #define segmap_group_sizze_1827 (mainzisegmap_group_sizze_1826)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2404;\n    int32_t local_tid_2405;\n    int32_t group_sizze_2408;\n    int32_t wave_sizze_2407;\n    int32_t group_tid_2406;\n    \n    global_tid_2404 = get_global_id(0);\n    local_tid_2405 = get_local_id(0);\n    group_sizze_2408 = get_local_size(0);\n    wave_sizze_2407 = LOCKSTEP_WIDTH;\n    group_tid_2406 = get_group_id(0);\n    \n    int32_t phys_tid_1823 = global_tid_2404;\n    int32_t gtid_1822 = add32(mul32(group_tid_2406, segmap_group_sizze_1827),\n                              local_tid_2405);\n    \n    if (slt32(gtid_1822, res_943)) {\n        int32_t x_1834 = ((__global int32_t *) res_mem_2045)[gtid_1822];\n        bool x_1835 = sle32(0, x_1834);\n        bool y_1836 = slt32(x_1834, res_943);\n        bool bounds_check_1837 = x_1835 && y_1836;\n        bool index_concat_cmp_183",
            "8 = sle32(m_911, x_1834);\n        \n        ((__global bool *) mem_2051)[gtid_1822] = bounds_check_1837;\n        ((__global bool *) mem_2053)[gtid_1822] = index_concat_cmp_1838;\n    }\n    \n  error_0:\n    return;\n    #undef segmap_group_sizze_1827\n}\n__kernel void segred_large_1372(__global int *global_failure,\n                                uint sync_arr_mem_2155_backing_offset_0,\n                                uint red_arr_mem_2153_backing_offset_1,\n                                int32_t m_911, int32_t d_912,\n                                int32_t num_groups_1369, __global\n                                unsigned char *mem_1909, __global\n                                unsigned char *mem_1913,\n                                int32_t vit_num_groups_2141,\n                                int32_t thread_per_segment_2143, __global\n                                unsigned char *group_res_arr_mem_2144, __global\n                                unsigned char *counter_mem_2146)\n{\n    #define segred_group_sizze_1367 (mainzisegred_group_sizze_1366)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *sync_arr_mem_2155_backing_1 =\n                  &shared_mem[sync_arr_mem_2155_backing_offset_0];\n    volatile char *red_arr_mem_2153_backing_0 =\n                  &shared_mem[red_arr_mem_2153_backing_offset_1];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2148;\n    int32_t local_tid_2149;\n    int32_t group_sizze_2152;\n    int32_t wave_sizze_2151;\n    int32_t group_tid_2150;\n    \n    global_tid_2148 = get_global_id(0);\n    local_tid_2149 = get_local_id(0);\n    group_sizze_2152 = get_local_size(0);\n    wave_sizze_2151 = LOCKSTEP_WIDTH;\n    group_tid_2150 = get_group_id(0);\n    \n    int32_t phys_tid_1372 = global_tid_2148;\n    __local char *red_arr_mem_2153;\n    \n    red_arr_mem_2153 = (__local char *) red_arr_mem_2153_backing_0;\n    \n    __local char *sync_arr_mem_2155;\n    \n    sync_arr",
            "_mem_2155 = (__local char *) sync_arr_mem_2155_backing_1;\n    \n    int32_t phys_group_id_2157;\n    \n    phys_group_id_2157 = get_group_id(0);\n    for (int32_t i_2158 = 0; i_2158 <\n         squot32(sub32(add32(sub32(vit_num_groups_2141, phys_group_id_2157),\n                             num_groups_1369), 1), num_groups_1369); i_2158++) {\n        int32_t virt_group_id_2159 = add32(phys_group_id_2157, mul32(i_2158,\n                                                                     num_groups_1369));\n        int32_t gtid_1359 = squot32(virt_group_id_2159,\n                                    squot32(sub32(add32(num_groups_1369,\n                                                        smax32(1, d_912)), 1),\n                                            smax32(1, d_912)));\n        int32_t gtid_1371;\n        float x_acc_2160;\n        int32_t chunk_sizze_2161 = smin32(squot32(sub32(add32(m_911,\n                                                              mul32(segred_group_sizze_1367,\n                                                                    squot32(sub32(add32(num_groups_1369,\n                                                                                        smax32(1,\n                                                                                               d_912)),\n                                                                                  1),\n                                                                            smax32(1,\n                                                                                   d_912)))),\n                                                        1),\n                                                  mul32(segred_group_sizze_1367,\n                                                        squot32(sub32(add32(num_groups_1369,\n                                                                            smax32(1,\n                                                                                   d_912)),\n                         ",
            "                                             1),\n                                                                smax32(1,\n                                                                       d_912)))),\n                                          squot32(sub32(add32(sub32(m_911,\n                                                                    srem32(add32(mul32(virt_group_id_2159,\n                                                                                       segred_group_sizze_1367),\n                                                                                 local_tid_2149),\n                                                                           mul32(segred_group_sizze_1367,\n                                                                                 squot32(sub32(add32(num_groups_1369,\n                                                                                                     smax32(1,\n                                                                                                            d_912)),\n                                                                                               1),\n                                                                                         smax32(1,\n                                                                                                d_912))))),\n                                                              thread_per_segment_2143),\n                                                        1),\n                                                  thread_per_segment_2143));\n        float x_1373;\n        float x_1374;\n        \n        // neutral-initialise the accumulators\n        {\n            x_acc_2160 = INFINITY;\n        }\n        for (int32_t i_2165 = 0; i_2165 < chunk_sizze_2161; i_2165++) {\n            gtid_1371 = add32(srem32(add32(mul32(virt_group_id_2159,\n                                                 segred_group_sizze_1367),\n                                           local",
            "_tid_2149),\n                                     mul32(segred_group_sizze_1367,\n                                           squot32(sub32(add32(num_groups_1369,\n                                                               smax32(1,\n                                                                      d_912)),\n                                                         1), smax32(1,\n                                                                    d_912)))),\n                              mul32(thread_per_segment_2143, i_2165));\n            // apply map function\n            {\n                float x_1377 = ((__global\n                                 float *) mem_1909)[add32(mul32(gtid_1359,\n                                                                m_911),\n                                                          gtid_1371)];\n                \n                // save map-out results\n                { }\n                // load accumulator\n                {\n                    x_1373 = x_acc_2160;\n                }\n                // load new values\n                {\n                    x_1374 = x_1377;\n                }\n                // apply reduction operator\n                {\n                    float res_1375 = fmin32(x_1373, x_1374);\n                    \n                    // store in accumulator\n                    {\n                        x_acc_2160 = res_1375;\n                    }\n                }\n            }\n        }\n        // to reduce current chunk, first store our result in memory\n        {\n            x_1373 = x_acc_2160;\n            ((__local float *) red_arr_mem_2153)[local_tid_2149] = x_1373;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        int32_t offset_2166;\n        int32_t skip_waves_2167;\n        float x_2162;\n        float x_2163;\n        \n        offset_2166 = 0;\n        // participating threads read initial accumulator\n        {\n            if (slt32(local_tid_2149, segred_group_sizze_1367)) {\n                x_2162",
            " = ((__local\n                           float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                            offset_2166)];\n            }\n        }\n        offset_2166 = 1;\n        while (slt32(offset_2166, wave_sizze_2151)) {\n            if (slt32(add32(local_tid_2149, offset_2166),\n                      segred_group_sizze_1367) && (sub32(local_tid_2149,\n                                                         mul32(squot32(local_tid_2149,\n                                                                       wave_sizze_2151),\n                                                               wave_sizze_2151)) &\n                                                   sub32(mul32(2, offset_2166),\n                                                         1)) == 0) {\n                // read array element\n                {\n                    x_2163 = ((volatile __local\n                               float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                                offset_2166)];\n                }\n                // apply reduction operation\n                {\n                    float res_2164 = fmin32(x_2162, x_2163);\n                    \n                    x_2162 = res_2164;\n                }\n                // write result of operation\n                {\n                    ((volatile __local\n                      float *) red_arr_mem_2153)[local_tid_2149] = x_2162;\n                }\n            }\n            offset_2166 *= 2;\n        }\n        skip_waves_2167 = 1;\n        while (slt32(skip_waves_2167,\n                     squot32(sub32(add32(segred_group_sizze_1367,\n                                         wave_sizze_2151), 1),\n                             wave_sizze_2151))) {\n            barrier(CLK_LOCAL_MEM_FENCE);\n            offset_2166 = mul32(skip_waves_2167, wave_sizze_2151);\n            if (slt32(add32(local_tid_2149, offset_2166),\n                      segred_group_sizze_1367) &&",
            " (sub32(local_tid_2149,\n                                                         mul32(squot32(local_tid_2149,\n                                                                       wave_sizze_2151),\n                                                               wave_sizze_2151)) ==\n                                                   0 && (squot32(local_tid_2149,\n                                                                 wave_sizze_2151) &\n                                                         sub32(mul32(2,\n                                                                     skip_waves_2167),\n                                                               1)) == 0)) {\n                // read array element\n                {\n                    x_2163 = ((__local\n                               float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                                offset_2166)];\n                }\n                // apply reduction operation\n                {\n                    float res_2164 = fmin32(x_2162, x_2163);\n                    \n                    x_2162 = res_2164;\n                }\n                // write result of operation\n                {\n                    ((__local float *) red_arr_mem_2153)[local_tid_2149] =\n                        x_2162;\n                }\n            }\n            skip_waves_2167 *= 2;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread saves the result in accumulator\n        {\n            if (local_tid_2149 == 0) {\n                x_acc_2160 = x_2162;\n            }\n        }\n        if (squot32(sub32(add32(num_groups_1369, smax32(1, d_912)), 1),\n                    smax32(1, d_912)) == 1) {\n            // first thread in group saves final result to memory\n            {\n                if (local_tid_2149 == 0) {\n                    ((__global float *) mem_1913)[gtid_1359] = x_acc_2160;\n                }\n            }\n        } else {\n            int32_t old_cou",
            "nter_2168;\n            \n            // first thread in group saves group result to global memory\n            {\n                if (local_tid_2149 == 0) {\n                    ((__global\n                      float *) group_res_arr_mem_2144)[mul32(virt_group_id_2159,\n                                                             segred_group_sizze_1367)] =\n                        x_acc_2160;\n                    mem_fence_global();\n                    old_counter_2168 =\n                        atomic_add_i32_global(&((volatile __global\n                                                 int *) counter_mem_2146)[srem32(squot32(virt_group_id_2159,\n                                                                                         squot32(sub32(add32(num_groups_1369,\n                                                                                                             smax32(1,\n                                                                                                                    d_912)),\n                                                                                                       1),\n                                                                                                 smax32(1,\n                                                                                                        d_912))),\n                                                                                 10240)],\n                                              (int) 1);\n                    ((__local bool *) sync_arr_mem_2155)[0] =\n                        old_counter_2168 ==\n                        sub32(squot32(sub32(add32(num_groups_1369, smax32(1,\n                                                                          d_912)),\n                                            1), smax32(1, d_912)), 1);\n                }\n            }\n            barrier(CLK_GLOBAL_MEM_FENCE);\n            \n            bool is_last_group_2169 = ((__local bool *) sync_arr_mem_2155)[0];\n      ",
            "      \n            if (is_last_group_2169) {\n                if (local_tid_2149 == 0) {\n                    old_counter_2168 =\n                        atomic_add_i32_global(&((volatile __global\n                                                 int *) counter_mem_2146)[srem32(squot32(virt_group_id_2159,\n                                                                                         squot32(sub32(add32(num_groups_1369,\n                                                                                                             smax32(1,\n                                                                                                                    d_912)),\n                                                                                                       1),\n                                                                                                 smax32(1,\n                                                                                                        d_912))),\n                                                                                 10240)],\n                                              (int) sub32(0,\n                                                          squot32(sub32(add32(num_groups_1369,\n                                                                              smax32(1,\n                                                                                     d_912)),\n                                                                        1),\n                                                                  smax32(1,\n                                                                         d_912))));\n                }\n                // read in the per-group-results\n                {\n                    if (slt32(local_tid_2149,\n                              squot32(sub32(add32(num_groups_1369, smax32(1,\n                                                                          d_912)),\n                                            ",
            "1), smax32(1, d_912)))) {\n                        x_1373 = ((__global\n                                   float *) group_res_arr_mem_2144)[mul32(add32(mul32(squot32(virt_group_id_2159,\n                                                                                              squot32(sub32(add32(num_groups_1369,\n                                                                                                                  smax32(1,\n                                                                                                                         d_912)),\n                                                                                                            1),\n                                                                                                      smax32(1,\n                                                                                                             d_912))),\n                                                                                      squot32(sub32(add32(num_groups_1369,\n                                                                                                          smax32(1,\n                                                                                                                 d_912)),\n                                                                                                    1),\n                                                                                              smax32(1,\n                                                                                                     d_912))),\n                                                                                local_tid_2149),\n                                                                          segred_group_sizze_1367)];\n                    } else {\n                        x_1373 = INFINITY;\n                    }\n                    ((__local float *) red_arr_mem_2153)[local_tid_2149] =\n                        x_1373;\n         ",
            "       }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // reduce the per-group results\n                {\n                    int32_t offset_2170;\n                    int32_t skip_waves_2171;\n                    float x_2162;\n                    float x_2163;\n                    \n                    offset_2170 = 0;\n                    // participating threads read initial accumulator\n                    {\n                        if (slt32(local_tid_2149, segred_group_sizze_1367)) {\n                            x_2162 = ((__local\n                                       float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                                        offset_2170)];\n                        }\n                    }\n                    offset_2170 = 1;\n                    while (slt32(offset_2170, wave_sizze_2151)) {\n                        if (slt32(add32(local_tid_2149, offset_2170),\n                                  segred_group_sizze_1367) &&\n                            (sub32(local_tid_2149, mul32(squot32(local_tid_2149,\n                                                                 wave_sizze_2151),\n                                                         wave_sizze_2151)) &\n                             sub32(mul32(2, offset_2170), 1)) == 0) {\n                            // read array element\n                            {\n                                x_2163 = ((volatile __local\n                                           float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                                            offset_2170)];\n                            }\n                            // apply reduction operation\n                            {\n                                float res_2164 = fmin32(x_2162, x_2163);\n                                \n                                x_2162 = res_2164;\n                            }\n                            // write result of operati",
            "on\n                            {\n                                ((volatile __local\n                                  float *) red_arr_mem_2153)[local_tid_2149] =\n                                    x_2162;\n                            }\n                        }\n                        offset_2170 *= 2;\n                    }\n                    skip_waves_2171 = 1;\n                    while (slt32(skip_waves_2171,\n                                 squot32(sub32(add32(segred_group_sizze_1367,\n                                                     wave_sizze_2151), 1),\n                                         wave_sizze_2151))) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                        offset_2170 = mul32(skip_waves_2171, wave_sizze_2151);\n                        if (slt32(add32(local_tid_2149, offset_2170),\n                                  segred_group_sizze_1367) &&\n                            (sub32(local_tid_2149, mul32(squot32(local_tid_2149,\n                                                                 wave_sizze_2151),\n                                                         wave_sizze_2151)) ==\n                             0 && (squot32(local_tid_2149, wave_sizze_2151) &\n                                   sub32(mul32(2, skip_waves_2171), 1)) == 0)) {\n                            // read array element\n                            {\n                                x_2163 = ((__local\n                                           float *) red_arr_mem_2153)[add32(local_tid_2149,\n                                                                            offset_2170)];\n                            }\n                            // apply reduction operation\n                            {\n                                float res_2164 = fmin32(x_2162, x_2163);\n                                \n                                x_2162 = res_2164;\n                            }\n                            // write result of operation\n                          ",
            "  {\n                                ((__local\n                                  float *) red_arr_mem_2153)[local_tid_2149] =\n                                    x_2162;\n                            }\n                        }\n                        skip_waves_2171 *= 2;\n                    }\n                    // and back to memory with the final result\n                    {\n                        if (local_tid_2149 == 0) {\n                            ((__global float *) mem_1913)[gtid_1359] = x_2162;\n                        }\n                    }\n                }\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE);\n    }\n    \n  error_1:\n    return;\n    #undef segred_group_sizze_1367\n}\n__kernel void segred_large_1391(__global int *global_failure,\n                                uint sync_arr_mem_2207_backing_offset_0,\n                                uint red_arr_mem_2205_backing_offset_1,\n                                int32_t m_911, int32_t d_912,\n                                int32_t num_groups_1388, __global\n                                unsigned char *mem_1909, __global\n                                unsigned char *mem_1917,\n                                int32_t vit_num_groups_2193,\n                                int32_t thread_per_segment_2195, __global\n                                unsigned char *group_res_arr_mem_2196, __global\n                                unsigned char *counter_mem_2198)\n{\n    #define segred_group_sizze_1386 (mainzisegred_group_sizze_1385)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *sync_arr_mem_2207_backing_1 =\n                  &shared_mem[sync_arr_mem_2207_backing_offset_0];\n    volatile char *red_arr_mem_2205_backing_0 =\n                  &shared_mem[red_arr_mem_2205_backing_offset_1];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2200;\n    int32_t local_tid_2201;\n    int32_t group_sizze_2204;\n    int32_t wav",
            "e_sizze_2203;\n    int32_t group_tid_2202;\n    \n    global_tid_2200 = get_global_id(0);\n    local_tid_2201 = get_local_id(0);\n    group_sizze_2204 = get_local_size(0);\n    wave_sizze_2203 = LOCKSTEP_WIDTH;\n    group_tid_2202 = get_group_id(0);\n    \n    int32_t phys_tid_1391 = global_tid_2200;\n    __local char *red_arr_mem_2205;\n    \n    red_arr_mem_2205 = (__local char *) red_arr_mem_2205_backing_0;\n    \n    __local char *sync_arr_mem_2207;\n    \n    sync_arr_mem_2207 = (__local char *) sync_arr_mem_2207_backing_1;\n    \n    int32_t phys_group_id_2209;\n    \n    phys_group_id_2209 = get_group_id(0);\n    for (int32_t i_2210 = 0; i_2210 <\n         squot32(sub32(add32(sub32(vit_num_groups_2193, phys_group_id_2209),\n                             num_groups_1388), 1), num_groups_1388); i_2210++) {\n        int32_t virt_group_id_2211 = add32(phys_group_id_2209, mul32(i_2210,\n                                                                     num_groups_1388));\n        int32_t gtid_1378 = squot32(virt_group_id_2211,\n                                    squot32(sub32(add32(num_groups_1388,\n                                                        smax32(1, d_912)), 1),\n                                            smax32(1, d_912)));\n        int32_t gtid_1390;\n        float x_acc_2212;\n        int32_t chunk_sizze_2213 = smin32(squot32(sub32(add32(m_911,\n                                                              mul32(segred_group_sizze_1386,\n                                                                    squot32(sub32(add32(num_groups_1388,\n                                                                                        smax32(1,\n                                                                                               d_912)),\n                                                                                  1),\n                                                                            smax32(1,\n                                                                      ",
            "             d_912)))),\n                                                        1),\n                                                  mul32(segred_group_sizze_1386,\n                                                        squot32(sub32(add32(num_groups_1388,\n                                                                            smax32(1,\n                                                                                   d_912)),\n                                                                      1),\n                                                                smax32(1,\n                                                                       d_912)))),\n                                          squot32(sub32(add32(sub32(m_911,\n                                                                    srem32(add32(mul32(virt_group_id_2211,\n                                                                                       segred_group_sizze_1386),\n                                                                                 local_tid_2201),\n                                                                           mul32(segred_group_sizze_1386,\n                                                                                 squot32(sub32(add32(num_groups_1388,\n                                                                                                     smax32(1,\n                                                                                                            d_912)),\n                                                                                               1),\n                                                                                         smax32(1,\n                                                                                                d_912))))),\n                                                              thread_per_segment_2195),\n                                                        1),\n                                   ",
            "               thread_per_segment_2195));\n        float x_1392;\n        float x_1393;\n        \n        // neutral-initialise the accumulators\n        {\n            x_acc_2212 = -INFINITY;\n        }\n        for (int32_t i_2217 = 0; i_2217 < chunk_sizze_2213; i_2217++) {\n            gtid_1390 = add32(srem32(add32(mul32(virt_group_id_2211,\n                                                 segred_group_sizze_1386),\n                                           local_tid_2201),\n                                     mul32(segred_group_sizze_1386,\n                                           squot32(sub32(add32(num_groups_1388,\n                                                               smax32(1,\n                                                                      d_912)),\n                                                         1), smax32(1,\n                                                                    d_912)))),\n                              mul32(thread_per_segment_2195, i_2217));\n            // apply map function\n            {\n                float x_1396 = ((__global\n                                 float *) mem_1909)[add32(mul32(gtid_1378,\n                                                                m_911),\n                                                          gtid_1390)];\n                \n                // save map-out results\n                { }\n                // load accumulator\n                {\n                    x_1392 = x_acc_2212;\n                }\n                // load new values\n                {\n                    x_1393 = x_1396;\n                }\n                // apply reduction operator\n                {\n                    float res_1394 = fmax32(x_1392, x_1393);\n                    \n                    // store in accumulator\n                    {\n                        x_acc_2212 = res_1394;\n                    }\n                }\n            }\n        }\n        // to reduce current chunk, first store our result in memory\n        {",
            "\n            x_1392 = x_acc_2212;\n            ((__local float *) red_arr_mem_2205)[local_tid_2201] = x_1392;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        int32_t offset_2218;\n        int32_t skip_waves_2219;\n        float x_2214;\n        float x_2215;\n        \n        offset_2218 = 0;\n        // participating threads read initial accumulator\n        {\n            if (slt32(local_tid_2201, segred_group_sizze_1386)) {\n                x_2214 = ((__local\n                           float *) red_arr_mem_2205)[add32(local_tid_2201,\n                                                            offset_2218)];\n            }\n        }\n        offset_2218 = 1;\n        while (slt32(offset_2218, wave_sizze_2203)) {\n            if (slt32(add32(local_tid_2201, offset_2218),\n                      segred_group_sizze_1386) && (sub32(local_tid_2201,\n                                                         mul32(squot32(local_tid_2201,\n                                                                       wave_sizze_2203),\n                                                               wave_sizze_2203)) &\n                                                   sub32(mul32(2, offset_2218),\n                                                         1)) == 0) {\n                // read array element\n                {\n                    x_2215 = ((volatile __local\n                               float *) red_arr_mem_2205)[add32(local_tid_2201,\n                                                                offset_2218)];\n                }\n                // apply reduction operation\n                {\n                    float res_2216 = fmax32(x_2214, x_2215);\n                    \n                    x_2214 = res_2216;\n                }\n                // write result of operation\n                {\n                    ((volatile __local\n                      float *) red_arr_mem_2205)[local_tid_2201] = x_2214;\n                }\n            }\n            offset_2218 *= 2;\n        }\n",
            "        skip_waves_2219 = 1;\n        while (slt32(skip_waves_2219,\n                     squot32(sub32(add32(segred_group_sizze_1386,\n                                         wave_sizze_2203), 1),\n                             wave_sizze_2203))) {\n            barrier(CLK_LOCAL_MEM_FENCE);\n            offset_2218 = mul32(skip_waves_2219, wave_sizze_2203);\n            if (slt32(add32(local_tid_2201, offset_2218),\n                      segred_group_sizze_1386) && (sub32(local_tid_2201,\n                                                         mul32(squot32(local_tid_2201,\n                                                                       wave_sizze_2203),\n                                                               wave_sizze_2203)) ==\n                                                   0 && (squot32(local_tid_2201,\n                                                                 wave_sizze_2203) &\n                                                         sub32(mul32(2,\n                                                                     skip_waves_2219),\n                                                               1)) == 0)) {\n                // read array element\n                {\n                    x_2215 = ((__local\n                               float *) red_arr_mem_2205)[add32(local_tid_2201,\n                                                                offset_2218)];\n                }\n                // apply reduction operation\n                {\n                    float res_2216 = fmax32(x_2214, x_2215);\n                    \n                    x_2214 = res_2216;\n                }\n                // write result of operation\n                {\n                    ((__local float *) red_arr_mem_2205)[local_tid_2201] =\n                        x_2214;\n                }\n            }\n            skip_waves_2219 *= 2;\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // first thread saves the result in accumulator\n        {\n            if (local_tid_2201",
            " == 0) {\n                x_acc_2212 = x_2214;\n            }\n        }\n        if (squot32(sub32(add32(num_groups_1388, smax32(1, d_912)), 1),\n                    smax32(1, d_912)) == 1) {\n            // first thread in group saves final result to memory\n            {\n                if (local_tid_2201 == 0) {\n                    ((__global float *) mem_1917)[gtid_1378] = x_acc_2212;\n                }\n            }\n        } else {\n            int32_t old_counter_2220;\n            \n            // first thread in group saves group result to global memory\n            {\n                if (local_tid_2201 == 0) {\n                    ((__global\n                      float *) group_res_arr_mem_2196)[mul32(virt_group_id_2211,\n                                                             segred_group_sizze_1386)] =\n                        x_acc_2212;\n                    mem_fence_global();\n                    old_counter_2220 =\n                        atomic_add_i32_global(&((volatile __global\n                                                 int *) counter_mem_2198)[srem32(squot32(virt_group_id_2211,\n                                                                                         squot32(sub32(add32(num_groups_1388,\n                                                                                                             smax32(1,\n                                                                                                                    d_912)),\n                                                                                                       1),\n                                                                                                 smax32(1,\n                                                                                                        d_912))),\n                                                                                 10240)],\n                                              (int) 1);\n                    ((__local bool *) sync_arr_m",
            "em_2207)[0] =\n                        old_counter_2220 ==\n                        sub32(squot32(sub32(add32(num_groups_1388, smax32(1,\n                                                                          d_912)),\n                                            1), smax32(1, d_912)), 1);\n                }\n            }\n            barrier(CLK_GLOBAL_MEM_FENCE);\n            \n            bool is_last_group_2221 = ((__local bool *) sync_arr_mem_2207)[0];\n            \n            if (is_last_group_2221) {\n                if (local_tid_2201 == 0) {\n                    old_counter_2220 =\n                        atomic_add_i32_global(&((volatile __global\n                                                 int *) counter_mem_2198)[srem32(squot32(virt_group_id_2211,\n                                                                                         squot32(sub32(add32(num_groups_1388,\n                                                                                                             smax32(1,\n                                                                                                                    d_912)),\n                                                                                                       1),\n                                                                                                 smax32(1,\n                                                                                                        d_912))),\n                                                                                 10240)],\n                                              (int) sub32(0,\n                                                          squot32(sub32(add32(num_groups_1388,\n                                                                              smax32(1,\n                                                                                     d_912)),\n                                                                        1),\n                                 ",
            "                                 smax32(1,\n                                                                         d_912))));\n                }\n                // read in the per-group-results\n                {\n                    if (slt32(local_tid_2201,\n                              squot32(sub32(add32(num_groups_1388, smax32(1,\n                                                                          d_912)),\n                                            1), smax32(1, d_912)))) {\n                        x_1392 = ((__global\n                                   float *) group_res_arr_mem_2196)[mul32(add32(mul32(squot32(virt_group_id_2211,\n                                                                                              squot32(sub32(add32(num_groups_1388,\n                                                                                                                  smax32(1,\n                                                                                                                         d_912)),\n                                                                                                            1),\n                                                                                                      smax32(1,\n                                                                                                             d_912))),\n                                                                                      squot32(sub32(add32(num_groups_1388,\n                                                                                                          smax32(1,\n                                                                                                                 d_912)),\n                                                                                                    1),\n                                                                                              smax32(1,\n                                                         ",
            "                                            d_912))),\n                                                                                local_tid_2201),\n                                                                          segred_group_sizze_1386)];\n                    } else {\n                        x_1392 = -INFINITY;\n                    }\n                    ((__local float *) red_arr_mem_2205)[local_tid_2201] =\n                        x_1392;\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // reduce the per-group results\n                {\n                    int32_t offset_2222;\n                    int32_t skip_waves_2223;\n                    float x_2214;\n                    float x_2215;\n                    \n                    offset_2222 = 0;\n                    // participating threads read initial accumulator\n                    {\n                        if (slt32(local_tid_2201, segred_group_sizze_1386)) {\n                            x_2214 = ((__local\n                                       float *) red_arr_mem_2205)[add32(local_tid_2201,\n                                                                        offset_2222)];\n                        }\n                    }\n                    offset_2222 = 1;\n                    while (slt32(offset_2222, wave_sizze_2203)) {\n                        if (slt32(add32(local_tid_2201, offset_2222),\n                                  segred_group_sizze_1386) &&\n                            (sub32(local_tid_2201, mul32(squot32(local_tid_2201,\n                                                                 wave_sizze_2203),\n                                                         wave_sizze_2203)) &\n                             sub32(mul32(2, offset_2222), 1)) == 0) {\n                            // read array element\n                            {\n                                x_2215 = ((volatile __local\n                                           float *) red_arr_mem_2205)[add32(lo",
            "cal_tid_2201,\n                                                                            offset_2222)];\n                            }\n                            // apply reduction operation\n                            {\n                                float res_2216 = fmax32(x_2214, x_2215);\n                                \n                                x_2214 = res_2216;\n                            }\n                            // write result of operation\n                            {\n                                ((volatile __local\n                                  float *) red_arr_mem_2205)[local_tid_2201] =\n                                    x_2214;\n                            }\n                        }\n                        offset_2222 *= 2;\n                    }\n                    skip_waves_2223 = 1;\n                    while (slt32(skip_waves_2223,\n                                 squot32(sub32(add32(segred_group_sizze_1386,\n                                                     wave_sizze_2203), 1),\n                                         wave_sizze_2203))) {\n                        barrier(CLK_LOCAL_MEM_FENCE);\n                        offset_2222 = mul32(skip_waves_2223, wave_sizze_2203);\n                        if (slt32(add32(local_tid_2201, offset_2222),\n                                  segred_group_sizze_1386) &&\n                            (sub32(local_tid_2201, mul32(squot32(local_tid_2201,\n                                                                 wave_sizze_2203),\n                                                         wave_sizze_2203)) ==\n                             0 && (squot32(local_tid_2201, wave_sizze_2203) &\n                                   sub32(mul32(2, skip_waves_2223), 1)) == 0)) {\n                            // read array element\n                            {\n                                x_2215 = ((__local\n                                           float *) red_arr_mem_2205)[add32(local_tid_2201,\n               ",
            "                                                             offset_2222)];\n                            }\n                            // apply reduction operation\n                            {\n                                float res_2216 = fmax32(x_2214, x_2215);\n                                \n                                x_2214 = res_2216;\n                            }\n                            // write result of operation\n                            {\n                                ((__local\n                                  float *) red_arr_mem_2205)[local_tid_2201] =\n                                    x_2214;\n                            }\n                        }\n                        skip_waves_2223 *= 2;\n                    }\n                    // and back to memory with the final result\n                    {\n                        if (local_tid_2201 == 0) {\n                            ((__global float *) mem_1917)[gtid_1378] = x_2214;\n                        }\n                    }\n                }\n            }\n        }\n        barrier(CLK_GLOBAL_MEM_FENCE);\n    }\n    \n  error_1:\n    return;\n    #undef segred_group_sizze_1386\n}\n__kernel void segred_small_1372(__global int *global_failure,\n                                uint red_arr_mem_2128_backing_offset_0,\n                                int32_t m_911, int32_t d_912,\n                                int32_t num_groups_1369, __global\n                                unsigned char *mem_1909, __global\n                                unsigned char *mem_1913,\n                                int32_t segment_sizze_nonzzero_2121)\n{\n    #define segred_group_sizze_1367 (mainzisegred_group_sizze_1366)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *red_arr_mem_2128_backing_0 =\n                  &shared_mem[red_arr_mem_2128_backing_offset_0];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2123;\n    int32_",
            "t local_tid_2124;\n    int32_t group_sizze_2127;\n    int32_t wave_sizze_2126;\n    int32_t group_tid_2125;\n    \n    global_tid_2123 = get_global_id(0);\n    local_tid_2124 = get_local_id(0);\n    group_sizze_2127 = get_local_size(0);\n    wave_sizze_2126 = LOCKSTEP_WIDTH;\n    group_tid_2125 = get_group_id(0);\n    \n    int32_t phys_tid_1372 = global_tid_2123;\n    __local char *red_arr_mem_2128;\n    \n    red_arr_mem_2128 = (__local char *) red_arr_mem_2128_backing_0;\n    \n    int32_t phys_group_id_2130;\n    \n    phys_group_id_2130 = get_group_id(0);\n    for (int32_t i_2131 = 0; i_2131 <\n         squot32(sub32(add32(sub32(squot32(sub32(add32(d_912,\n                                                       squot32(segred_group_sizze_1367,\n                                                               segment_sizze_nonzzero_2121)),\n                                                 1),\n                                           squot32(segred_group_sizze_1367,\n                                                   segment_sizze_nonzzero_2121)),\n                                   phys_group_id_2130), num_groups_1369), 1),\n                 num_groups_1369); i_2131++) {\n        int32_t virt_group_id_2132 = add32(phys_group_id_2130, mul32(i_2131,\n                                                                     num_groups_1369));\n        int32_t gtid_1359 = add32(squot32(local_tid_2124,\n                                          segment_sizze_nonzzero_2121),\n                                  mul32(virt_group_id_2132,\n                                        squot32(segred_group_sizze_1367,\n                                                segment_sizze_nonzzero_2121)));\n        int32_t gtid_1371 = srem32(local_tid_2124, m_911);\n        \n        // apply map function if in bounds\n        {\n            if (slt32(0, m_911) && (slt32(gtid_1359, d_912) &&\n                                    slt32(local_tid_2124, mul32(m_911,\n                                                                squot3",
            "2(segred_group_sizze_1367,\n                                                                        segment_sizze_nonzzero_2121))))) {\n                float x_1377 = ((__global\n                                 float *) mem_1909)[add32(mul32(gtid_1359,\n                                                                m_911),\n                                                          gtid_1371)];\n                \n                // save map-out results\n                { }\n                // save results to be reduced\n                {\n                    ((__local float *) red_arr_mem_2128)[local_tid_2124] =\n                        x_1377;\n                }\n            } else {\n                ((__local float *) red_arr_mem_2128)[local_tid_2124] = INFINITY;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(0, m_911)) {\n            // perform segmented scan to imitate reduction\n            {\n                float x_1373;\n                float x_1374;\n                float x_2133;\n                float x_2134;\n                int32_t skip_threads_2136;\n                \n                // read input for in-block scan\n                {\n                    if (slt32(local_tid_2124, mul32(m_911,\n                                                    squot32(segred_group_sizze_1367,\n                                                            segment_sizze_nonzzero_2121)))) {\n                        x_1374 = ((volatile __local\n                                   float *) red_arr_mem_2128)[local_tid_2124];\n                        if (sub32(local_tid_2124, mul32(squot32(local_tid_2124,\n                                                                32), 32)) ==\n                            0) {\n                            x_1373 = x_1374;\n                        }\n                    }\n                }\n                // in-block scan (hopefully no barriers needed)\n                {\n                    skip_threads_2136 = 1;\n                    while (slt3",
            "2(skip_threads_2136, 32)) {\n                        if (sle32(skip_threads_2136, sub32(local_tid_2124,\n                                                           mul32(squot32(local_tid_2124,\n                                                                         32),\n                                                                 32))) &&\n                            slt32(local_tid_2124, mul32(m_911,\n                                                        squot32(segred_group_sizze_1367,\n                                                                segment_sizze_nonzzero_2121)))) {\n                            // read operands\n                            {\n                                x_1373 = ((volatile __local\n                                           float *) red_arr_mem_2128)[sub32(local_tid_2124,\n                                                                            skip_threads_2136)];\n                            }\n                            // perform operation\n                            {\n                                bool inactive_2137 =\n                                     slt32(srem32(local_tid_2124, m_911),\n                                           sub32(local_tid_2124,\n                                                 sub32(local_tid_2124,\n                                                       skip_threads_2136)));\n                                \n                                if (inactive_2137) {\n                                    x_1373 = x_1374;\n                                }\n                                if (!inactive_2137) {\n                                    float res_1375 = fmin32(x_1373, x_1374);\n                                    \n                                    x_1373 = res_1375;\n                                }\n                            }\n                        }\n                        if (sle32(wave_sizze_2126, skip_threads_2136)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                ",
            "        }\n                        if (sle32(skip_threads_2136, sub32(local_tid_2124,\n                                                           mul32(squot32(local_tid_2124,\n                                                                         32),\n                                                                 32))) &&\n                            slt32(local_tid_2124, mul32(m_911,\n                                                        squot32(segred_group_sizze_1367,\n                                                                segment_sizze_nonzzero_2121)))) {\n                            // write result\n                            {\n                                ((volatile __local\n                                  float *) red_arr_mem_2128)[local_tid_2124] =\n                                    x_1373;\n                                x_1374 = x_1373;\n                            }\n                        }\n                        if (sle32(wave_sizze_2126, skip_threads_2136)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                        }\n                        skip_threads_2136 *= 2;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // last thread of block 'i' writes its result to offset 'i'\n                {\n                    if (sub32(local_tid_2124, mul32(squot32(local_tid_2124, 32),\n                                                    32)) == 31 &&\n                        slt32(local_tid_2124, mul32(m_911,\n                                                    squot32(segred_group_sizze_1367,\n                                                            segment_sizze_nonzzero_2121)))) {\n                        ((volatile __local\n                          float *) red_arr_mem_2128)[squot32(local_tid_2124,\n                                                             32)] = x_1373;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // ",
            "scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n                {\n                    int32_t skip_threads_2138;\n                    \n                    // read input for in-block scan\n                    {\n                        if (squot32(local_tid_2124, 32) == 0 &&\n                            slt32(local_tid_2124, mul32(m_911,\n                                                        squot32(segred_group_sizze_1367,\n                                                                segment_sizze_nonzzero_2121)))) {\n                            x_2134 = ((volatile __local\n                                       float *) red_arr_mem_2128)[local_tid_2124];\n                            if (sub32(local_tid_2124,\n                                      mul32(squot32(local_tid_2124, 32), 32)) ==\n                                0) {\n                                x_2133 = x_2134;\n                            }\n                        }\n                    }\n                    // in-block scan (hopefully no barriers needed)\n                    {\n                        skip_threads_2138 = 1;\n                        while (slt32(skip_threads_2138, 32)) {\n                            if (sle32(skip_threads_2138, sub32(local_tid_2124,\n                                                               mul32(squot32(local_tid_2124,\n                                                                             32),\n                                                                     32))) &&\n                                (squot32(local_tid_2124, 32) == 0 &&\n                                 slt32(local_tid_2124, mul32(m_911,\n                                                             squot32(segred_group_sizze_1367,\n                                                                     segment_sizze_nonzzero_2121))))) {\n                                // read operands\n                                {\n                                    x_2133 = ((volatile __l",
            "ocal\n                                               float *) red_arr_mem_2128)[sub32(local_tid_2124,\n                                                                                skip_threads_2138)];\n                                }\n                                // perform operation\n                                {\n                                    bool inactive_2139 =\n                                         slt32(srem32(sub32(add32(mul32(local_tid_2124,\n                                                                        32),\n                                                                  32), 1),\n                                                      m_911),\n                                               sub32(sub32(add32(mul32(local_tid_2124,\n                                                                       32), 32),\n                                                           1),\n                                                     sub32(add32(mul32(sub32(local_tid_2124,\n                                                                             skip_threads_2138),\n                                                                       32), 32),\n                                                           1)));\n                                    \n                                    if (inactive_2139) {\n                                        x_2133 = x_2134;\n                                    }\n                                    if (!inactive_2139) {\n                                        float res_2135 = fmin32(x_2133, x_2134);\n                                        \n                                        x_2133 = res_2135;\n                                    }\n                                }\n                            }\n                            if (sle32(wave_sizze_2126, skip_threads_2138)) {\n                                barrier(CLK_LOCAL_MEM_FENCE);\n                            }\n                            if (sle32(skip_threads_2",
            "138, sub32(local_tid_2124,\n                                                               mul32(squot32(local_tid_2124,\n                                                                             32),\n                                                                     32))) &&\n                                (squot32(local_tid_2124, 32) == 0 &&\n                                 slt32(local_tid_2124, mul32(m_911,\n                                                             squot32(segred_group_sizze_1367,\n                                                                     segment_sizze_nonzzero_2121))))) {\n                                // write result\n                                {\n                                    ((volatile __local\n                                      float *) red_arr_mem_2128)[local_tid_2124] =\n                                        x_2133;\n                                    x_2134 = x_2133;\n                                }\n                            }\n                            if (sle32(wave_sizze_2126, skip_threads_2138)) {\n                                barrier(CLK_LOCAL_MEM_FENCE);\n                            }\n                            skip_threads_2138 *= 2;\n                        }\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // carry-in for every block except the first\n                {\n                    if (!(squot32(local_tid_2124, 32) == 0 ||\n                          !slt32(local_tid_2124, mul32(m_911,\n                                                       squot32(segred_group_sizze_1367,\n                                                               segment_sizze_nonzzero_2121))))) {\n                        // read operands\n                        {\n                            x_1374 = x_1373;\n                            x_1373 = ((__local\n                                       float *) red_arr_mem_2128)[sub32(squot32(local_tid_2124,\n                       ",
            "                                                         32),\n                                                                        1)];\n                        }\n                        // perform operation\n                        {\n                            bool inactive_2140 = slt32(srem32(local_tid_2124,\n                                                              m_911),\n                                                       sub32(local_tid_2124,\n                                                             sub32(mul32(squot32(local_tid_2124,\n                                                                                 32),\n                                                                         32),\n                                                                   1)));\n                            \n                            if (inactive_2140) {\n                                x_1373 = x_1374;\n                            }\n                            if (!inactive_2140) {\n                                float res_1375 = fmin32(x_1373, x_1374);\n                                \n                                x_1373 = res_1375;\n                            }\n                        }\n                        // write final result\n                        {\n                            ((__local\n                              float *) red_arr_mem_2128)[local_tid_2124] =\n                                x_1373;\n                        }\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // restore correct values for first block\n                {\n                    if (squot32(local_tid_2124, 32) == 0) {\n                        ((__local float *) red_arr_mem_2128)[local_tid_2124] =\n                            x_1374;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // save final values of segments\n        ",
            "{\n            if (slt32(add32(mul32(virt_group_id_2132,\n                                  squot32(segred_group_sizze_1367,\n                                          segment_sizze_nonzzero_2121)),\n                            local_tid_2124), d_912) && slt32(local_tid_2124,\n                                                             squot32(segred_group_sizze_1367,\n                                                                     segment_sizze_nonzzero_2121))) {\n                ((__global float *) mem_1913)[add32(mul32(virt_group_id_2132,\n                                                          squot32(segred_group_sizze_1367,\n                                                                  segment_sizze_nonzzero_2121)),\n                                                    local_tid_2124)] = ((__local\n                                                                         float *) red_arr_mem_2128)[sub32(mul32(add32(local_tid_2124,\n                                                                                                                      1),\n                                                                                                                segment_sizze_nonzzero_2121),\n                                                                                                          1)];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        barrier(CLK_GLOBAL_MEM_FENCE);\n    }\n    \n  error_1:\n    return;\n    #undef segred_group_sizze_1367\n}\n__kernel void segred_small_1391(__global int *global_failure,\n                                uint red_arr_mem_2180_backing_offset_0,\n                                int32_t m_911, int32_t d_912,\n                                int32_t num_groups_1388, __global\n                                unsigned char *mem_1909, __global\n                                unsigned char *mem_1917,\n                                int32_t segment_sizze_nonzzero_2173)\n{\n    #define segred_group_sizze_1386 (mainzisegr",
            "ed_group_sizze_1385)\n    \n    const int block_dim0 = 0;\n    const int block_dim1 = 1;\n    const int block_dim2 = 2;\n    volatile char *red_arr_mem_2180_backing_0 =\n                  &shared_mem[red_arr_mem_2180_backing_offset_0];\n    \n    if (*global_failure >= 0)\n        return;\n    \n    int32_t global_tid_2175;\n    int32_t local_tid_2176;\n    int32_t group_sizze_2179;\n    int32_t wave_sizze_2178;\n    int32_t group_tid_2177;\n    \n    global_tid_2175 = get_global_id(0);\n    local_tid_2176 = get_local_id(0);\n    group_sizze_2179 = get_local_size(0);\n    wave_sizze_2178 = LOCKSTEP_WIDTH;\n    group_tid_2177 = get_group_id(0);\n    \n    int32_t phys_tid_1391 = global_tid_2175;\n    __local char *red_arr_mem_2180;\n    \n    red_arr_mem_2180 = (__local char *) red_arr_mem_2180_backing_0;\n    \n    int32_t phys_group_id_2182;\n    \n    phys_group_id_2182 = get_group_id(0);\n    for (int32_t i_2183 = 0; i_2183 <\n         squot32(sub32(add32(sub32(squot32(sub32(add32(d_912,\n                                                       squot32(segred_group_sizze_1386,\n                                                               segment_sizze_nonzzero_2173)),\n                                                 1),\n                                           squot32(segred_group_sizze_1386,\n                                                   segment_sizze_nonzzero_2173)),\n                                   phys_group_id_2182), num_groups_1388), 1),\n                 num_groups_1388); i_2183++) {\n        int32_t virt_group_id_2184 = add32(phys_group_id_2182, mul32(i_2183,\n                                                                     num_groups_1388));\n        int32_t gtid_1378 = add32(squot32(local_tid_2176,\n                                          segment_sizze_nonzzero_2173),\n                                  mul32(virt_group_id_2184,\n                                        squot32(segred_group_sizze_1386,\n                                                segment_sizze_nonzzero_2173)));\n",
            "        int32_t gtid_1390 = srem32(local_tid_2176, m_911);\n        \n        // apply map function if in bounds\n        {\n            if (slt32(0, m_911) && (slt32(gtid_1378, d_912) &&\n                                    slt32(local_tid_2176, mul32(m_911,\n                                                                squot32(segred_group_sizze_1386,\n                                                                        segment_sizze_nonzzero_2173))))) {\n                float x_1396 = ((__global\n                                 float *) mem_1909)[add32(mul32(gtid_1378,\n                                                                m_911),\n                                                          gtid_1390)];\n                \n                // save map-out results\n                { }\n                // save results to be reduced\n                {\n                    ((__local float *) red_arr_mem_2180)[local_tid_2176] =\n                        x_1396;\n                }\n            } else {\n                ((__local float *) red_arr_mem_2180)[local_tid_2176] =\n                    -INFINITY;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (slt32(0, m_911)) {\n            // perform segmented scan to imitate reduction\n            {\n                float x_1392;\n                float x_1393;\n                float x_2185;\n                float x_2186;\n                int32_t skip_threads_2188;\n                \n                // read input for in-block scan\n                {\n                    if (slt32(local_tid_2176, mul32(m_911,\n                                                    squot32(segred_group_sizze_1386,\n                                                            segment_sizze_nonzzero_2173)))) {\n                        x_1393 = ((volatile __local\n                                   float *) red_arr_mem_2180)[local_tid_2176];\n                        if (sub32(local_tid_2176, mul32(squot32(local_tid_2176,\n                               ",
            "                                 32), 32)) ==\n                            0) {\n                            x_1392 = x_1393;\n                        }\n                    }\n                }\n                // in-block scan (hopefully no barriers needed)\n                {\n                    skip_threads_2188 = 1;\n                    while (slt32(skip_threads_2188, 32)) {\n                        if (sle32(skip_threads_2188, sub32(local_tid_2176,\n                                                           mul32(squot32(local_tid_2176,\n                                                                         32),\n                                                                 32))) &&\n                            slt32(local_tid_2176, mul32(m_911,\n                                                        squot32(segred_group_sizze_1386,\n                                                                segment_sizze_nonzzero_2173)))) {\n                            // read operands\n                            {\n                                x_1392 = ((volatile __local\n                                           float *) red_arr_mem_2180)[sub32(local_tid_2176,\n                                                                            skip_threads_2188)];\n                            }\n                            // perform operation\n                            {\n                                bool inactive_2189 =\n                                     slt32(srem32(local_tid_2176, m_911),\n                                           sub32(local_tid_2176,\n                                                 sub32(local_tid_2176,\n                                                       skip_threads_2188)));\n                                \n                                if (inactive_2189) {\n                                    x_1392 = x_1393;\n                                }\n                                if (!inactive_2189) {\n                                    float res_1394 = fmax32(",
            "x_1392, x_1393);\n                                    \n                                    x_1392 = res_1394;\n                                }\n                            }\n                        }\n                        if (sle32(wave_sizze_2178, skip_threads_2188)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                        }\n                        if (sle32(skip_threads_2188, sub32(local_tid_2176,\n                                                           mul32(squot32(local_tid_2176,\n                                                                         32),\n                                                                 32))) &&\n                            slt32(local_tid_2176, mul32(m_911,\n                                                        squot32(segred_group_sizze_1386,\n                                                                segment_sizze_nonzzero_2173)))) {\n                            // write result\n                            {\n                                ((volatile __local\n                                  float *) red_arr_mem_2180)[local_tid_2176] =\n                                    x_1392;\n                                x_1393 = x_1392;\n                            }\n                        }\n                        if (sle32(wave_sizze_2178, skip_threads_2188)) {\n                            barrier(CLK_LOCAL_MEM_FENCE);\n                        }\n                        skip_threads_2188 *= 2;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // last thread of block 'i' writes its result to offset 'i'\n                {\n                    if (sub32(local_tid_2176, mul32(squot32(local_tid_2176, 32),\n                                                    32)) == 31 &&\n                        slt32(local_tid_2176, mul32(m_911,\n                                                    squot32(segred_group_sizze_1386,\n                                                 ",
            "           segment_sizze_nonzzero_2173)))) {\n                        ((volatile __local\n                          float *) red_arr_mem_2180)[squot32(local_tid_2176,\n                                                             32)] = x_1392;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // scan the first block, after which offset 'i' contains carry-in for block 'i+1'\n                {\n                    int32_t skip_threads_2190;\n                    \n                    // read input for in-block scan\n                    {\n                        if (squot32(local_tid_2176, 32) == 0 &&\n                            slt32(local_tid_2176, mul32(m_911,\n                                                        squot32(segred_group_sizze_1386,\n                                                                segment_sizze_nonzzero_2173)))) {\n                            x_2186 = ((volatile __local\n                                       float *) red_arr_mem_2180)[local_tid_2176];\n                            if (sub32(local_tid_2176,\n                                      mul32(squot32(local_tid_2176, 32), 32)) ==\n                                0) {\n                                x_2185 = x_2186;\n                            }\n                        }\n                    }\n                    // in-block scan (hopefully no barriers needed)\n                    {\n                        skip_threads_2190 = 1;\n                        while (slt32(skip_threads_2190, 32)) {\n                            if (sle32(skip_threads_2190, sub32(local_tid_2176,\n                                                               mul32(squot32(local_tid_2176,\n                                                                             32),\n                                                                     32))) &&\n                                (squot32(local_tid_2176, 32) == 0 &&\n                                 slt32(local_tid_2176, mul32(m",
            "_911,\n                                                             squot32(segred_group_sizze_1386,\n                                                                     segment_sizze_nonzzero_2173))))) {\n                                // read operands\n                                {\n                                    x_2185 = ((volatile __local\n                                               float *) red_arr_mem_2180)[sub32(local_tid_2176,\n                                                                                skip_threads_2190)];\n                                }\n                                // perform operation\n                                {\n                                    bool inactive_2191 =\n                                         slt32(srem32(sub32(add32(mul32(local_tid_2176,\n                                                                        32),\n                                                                  32), 1),\n                                                      m_911),\n                                               sub32(sub32(add32(mul32(local_tid_2176,\n                                                                       32), 32),\n                                                           1),\n                                                     sub32(add32(mul32(sub32(local_tid_2176,\n                                                                             skip_threads_2190),\n                                                                       32), 32),\n                                                           1)));\n                                    \n                                    if (inactive_2191) {\n                                        x_2185 = x_2186;\n                                    }\n                                    if (!inactive_2191) {\n                                        float res_2187 = fmax32(x_2185, x_2186);\n                                        \n                                    ",
            "    x_2185 = res_2187;\n                                    }\n                                }\n                            }\n                            if (sle32(wave_sizze_2178, skip_threads_2190)) {\n                                barrier(CLK_LOCAL_MEM_FENCE);\n                            }\n                            if (sle32(skip_threads_2190, sub32(local_tid_2176,\n                                                               mul32(squot32(local_tid_2176,\n                                                                             32),\n                                                                     32))) &&\n                                (squot32(local_tid_2176, 32) == 0 &&\n                                 slt32(local_tid_2176, mul32(m_911,\n                                                             squot32(segred_group_sizze_1386,\n                                                                     segment_sizze_nonzzero_2173))))) {\n                                // write result\n                                {\n                                    ((volatile __local\n                                      float *) red_arr_mem_2180)[local_tid_2176] =\n                                        x_2185;\n                                    x_2186 = x_2185;\n                                }\n                            }\n                            if (sle32(wave_sizze_2178, skip_threads_2190)) {\n                                barrier(CLK_LOCAL_MEM_FENCE);\n                            }\n                            skip_threads_2190 *= 2;\n                        }\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // carry-in for every block except the first\n                {\n                    if (!(squot32(local_tid_2176, 32) == 0 ||\n                          !slt32(local_tid_2176, mul32(m_911,\n                                                       squot32(segred_group_sizze_1386,\n                              ",
            "                                 segment_sizze_nonzzero_2173))))) {\n                        // read operands\n                        {\n                            x_1393 = x_1392;\n                            x_1392 = ((__local\n                                       float *) red_arr_mem_2180)[sub32(squot32(local_tid_2176,\n                                                                                32),\n                                                                        1)];\n                        }\n                        // perform operation\n                        {\n                            bool inactive_2192 = slt32(srem32(local_tid_2176,\n                                                              m_911),\n                                                       sub32(local_tid_2176,\n                                                             sub32(mul32(squot32(local_tid_2176,\n                                                                                 32),\n                                                                         32),\n                                                                   1)));\n                            \n                            if (inactive_2192) {\n                                x_1392 = x_1393;\n                            }\n                            if (!inactive_2192) {\n                                float res_1394 = fmax32(x_1392, x_1393);\n                                \n                                x_1392 = res_1394;\n                            }\n                        }\n                        // write final result\n                        {\n                            ((__local\n                              float *) red_arr_mem_2180)[local_tid_2176] =\n                                x_1392;\n                        }\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                // restore correct values for first block\n                {\n                    if (sq",
            "uot32(local_tid_2176, 32) == 0) {\n                        ((__local float *) red_arr_mem_2180)[local_tid_2176] =\n                            x_1393;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        // save final values of segments\n        {\n            if (slt32(add32(mul32(virt_group_id_2184,\n                                  squot32(segred_group_sizze_1386,\n                                          segment_sizze_nonzzero_2173)),\n                            local_tid_2176), d_912) && slt32(local_tid_2176,\n                                                             squot32(segred_group_sizze_1386,\n                                                                     segment_sizze_nonzzero_2173))) {\n                ((__global float *) mem_1917)[add32(mul32(virt_group_id_2184,\n                                                          squot32(segred_group_sizze_1386,\n                                                                  segment_sizze_nonzzero_2173)),\n                                                    local_tid_2176)] = ((__local\n                                                                         float *) red_arr_mem_2180)[sub32(mul32(add32(local_tid_2176,\n                                                                                                                      1),\n                                                                                                                segment_sizze_nonzzero_2173),\n                                                                                                          1)];\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        barrier(CLK_GLOBAL_MEM_FENCE);\n    }\n    \n  error_1:\n    return;\n    #undef segred_group_sizze_1386\n}\n",
            NULL};
static const char *size_names[] = {"main.group_size_2228",
                                   "main.group_size_2237",
                                   "main.group_size_2246",
                                   "main.group_size_2269",
                                   "main.group_size_2284",
                                   "main.segmap_group_size_1494",
                                   "main.segmap_group_size_1592",
                                   "main.segmap_group_size_1627",
                                   "main.segmap_group_size_1659",
                                   "main.segmap_group_size_1707",
                                   "main.segmap_group_size_1749",
                                   "main.segmap_group_size_1773",
                                   "main.segmap_group_size_1789",
                                   "main.segmap_group_size_1806",
                                   "main.segmap_group_size_1826",
                                   "main.segmap_num_groups_1661",
                                   "main.segred_group_size_1366",
                                   "main.segred_group_size_1385",
                                   "main.segred_num_groups_1368",
                                   "main.segred_num_groups_1387",
                                   "main.segscan_group_size_1683",
                                   "main.segscan_num_groups_1685"};
static const char *size_vars[] = {"mainzigroup_sizze_2228",
                                  "mainzigroup_sizze_2237",
                                  "mainzigroup_sizze_2246",
                                  "mainzigroup_sizze_2269",
                                  "mainzigroup_sizze_2284",
                                  "mainzisegmap_group_sizze_1494",
                                  "mainzisegmap_group_sizze_1592",
                                  "mainzisegmap_group_sizze_1627",
                                  "mainzisegmap_group_sizze_1659",
                                  "mainzisegmap_group_sizze_1707",
                                  "mainzisegmap_group_sizze_1749",
                                  "mainzisegmap_group_sizze_1773",
                                  "mainzisegmap_group_sizze_1789",
                                  "mainzisegmap_group_sizze_1806",
                                  "mainzisegmap_group_sizze_1826",
                                  "mainzisegmap_num_groups_1661",
                                  "mainzisegred_group_sizze_1366",
                                  "mainzisegred_group_sizze_1385",
                                  "mainzisegred_num_groups_1368",
                                  "mainzisegred_num_groups_1387",
                                  "mainzisegscan_group_sizze_1683",
                                  "mainzisegscan_num_groups_1685"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "num_groups", "group_size", "group_size",
                                     "num_groups", "num_groups", "group_size",
                                     "num_groups"};
int futhark_get_num_sizes(void)
{
    return 22;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
struct sizes {
    size_t mainzigroup_sizze_2228;
    size_t mainzigroup_sizze_2237;
    size_t mainzigroup_sizze_2246;
    size_t mainzigroup_sizze_2269;
    size_t mainzigroup_sizze_2284;
    size_t mainzisegmap_group_sizze_1494;
    size_t mainzisegmap_group_sizze_1592;
    size_t mainzisegmap_group_sizze_1627;
    size_t mainzisegmap_group_sizze_1659;
    size_t mainzisegmap_group_sizze_1707;
    size_t mainzisegmap_group_sizze_1749;
    size_t mainzisegmap_group_sizze_1773;
    size_t mainzisegmap_group_sizze_1789;
    size_t mainzisegmap_group_sizze_1806;
    size_t mainzisegmap_group_sizze_1826;
    size_t mainzisegmap_num_groups_1661;
    size_t mainzisegred_group_sizze_1366;
    size_t mainzisegred_group_sizze_1385;
    size_t mainzisegred_num_groups_1368;
    size_t mainzisegred_num_groups_1387;
    size_t mainzisegscan_group_sizze_1683;
    size_t mainzisegscan_num_groups_1685;
} ;
struct futhark_context_config {
    struct cuda_config cu_cfg;
    size_t sizes[22];
    int num_nvrtc_opts;
    const char **nvrtc_opts;
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->num_nvrtc_opts = 0;
    cfg->nvrtc_opts = (const char **) malloc(sizeof(const char *));
    cfg->nvrtc_opts[0] = NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    cfg->sizes[9] = 0;
    cfg->sizes[10] = 0;
    cfg->sizes[11] = 0;
    cfg->sizes[12] = 0;
    cfg->sizes[13] = 0;
    cfg->sizes[14] = 0;
    cfg->sizes[15] = 0;
    cfg->sizes[16] = 0;
    cfg->sizes[17] = 0;
    cfg->sizes[18] = 0;
    cfg->sizes[19] = 0;
    cfg->sizes[20] = 0;
    cfg->sizes[21] = 0;
    cuda_config_init(&cfg->cu_cfg, 22, size_names, size_vars, cfg->sizes,
                     size_classes);
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg->nvrtc_opts);
    free(cfg);
}
void futhark_context_config_add_nvrtc_option(struct futhark_context_config *cfg,
                                             const char *opt)
{
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = opt;
    cfg->num_nvrtc_opts++;
    cfg->nvrtc_opts = (const char **) realloc(cfg->nvrtc_opts,
                                              (cfg->num_nvrtc_opts + 1) *
                                              sizeof(const char *));
    cfg->nvrtc_opts[cfg->num_nvrtc_opts] = NULL;
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->cu_cfg.logging = cfg->cu_cfg.debugging = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->cu_cfg.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->cu_cfg, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->cu_cfg.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->cu_cfg.load_program_from = path;
}
void futhark_context_config_dump_ptx_to(struct futhark_context_config *cfg,
                                        const char *path)
{
    cfg->cu_cfg.dump_ptx_to = path;
}
void futhark_context_config_load_ptx_from(struct futhark_context_config *cfg,
                                          const char *path)
{
    cfg->cu_cfg.load_ptx_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->cu_cfg.default_block_size = size;
    cfg->cu_cfg.default_block_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->cu_cfg.default_grid_size = num;
    cfg->cu_cfg.default_grid_size_changed = 1;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_tile_size = size;
    cfg->cu_cfg.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->cu_cfg.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 22; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    if (strcmp(size_name, "default_group_size") == 0) {
        cfg->cu_cfg.default_block_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_num_groups") == 0) {
        cfg->cu_cfg.default_grid_size = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_threshold") == 0) {
        cfg->cu_cfg.default_threshold = size_value;
        return 0;
    }
    if (strcmp(size_name, "default_tile_size") == 0) {
        cfg->cu_cfg.default_tile_size = size_value;
        return 0;
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int profiling;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    struct { } constants;
    struct memblock_device counter_mem_2146;
    struct memblock_device counter_mem_2198;
    CUfunction iota_2225;
    CUfunction iota_2266;
    CUfunction map_transpose_f32;
    CUfunction map_transpose_f32_low_height;
    CUfunction map_transpose_f32_low_width;
    CUfunction map_transpose_f32_small;
    CUfunction map_transpose_i32;
    CUfunction map_transpose_i32_low_height;
    CUfunction map_transpose_i32_low_width;
    CUfunction map_transpose_i32_small;
    CUfunction replicate_2234;
    CUfunction replicate_2243;
    CUfunction replicate_2281;
    CUfunction scan_stage1_1689;
    CUfunction scan_stage2_1689;
    CUfunction scan_stage3_1689;
    CUfunction segmap_1491;
    CUfunction segmap_1587;
    CUfunction segmap_1622;
    CUfunction segmap_1656;
    CUfunction segmap_1702;
    CUfunction segmap_1744;
    CUfunction segmap_1770;
    CUfunction segmap_1786;
    CUfunction segmap_1801;
    CUfunction segmap_1823;
    CUfunction segred_large_1372;
    CUfunction segred_large_1391;
    CUfunction segred_small_1372;
    CUfunction segred_small_1391;
    CUdeviceptr global_failure;
    CUdeviceptr global_failure_args;
    struct cuda_context cuda;
    struct sizes sizes;
    int32_t failure_is_an_option;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->profiling = ctx->debugging = ctx->detail_memory =
        cfg->cu_cfg.debugging;
    ctx->error = NULL;
    ctx->cuda.cfg = cfg->cu_cfg;
    create_lock(&ctx->lock);
    ctx->failure_is_an_option = 0;
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    cuda_setup(&ctx->cuda, cuda_program, cfg->nvrtc_opts);
    
    int32_t no_error = -1;
    
    CUDA_SUCCEED(cuMemAlloc(&ctx->global_failure, sizeof(no_error)));
    CUDA_SUCCEED(cuMemcpyHtoD(ctx->global_failure, &no_error,
                              sizeof(no_error)));
    // The +1 is to avoid zero-byte allocations.
    CUDA_SUCCEED(cuMemAlloc(&ctx->global_failure_args, sizeof(int32_t) * (4 +
                                                                          1)));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->iota_2225, ctx->cuda.module,
                                     "iota_2225"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->iota_2266, ctx->cuda.module,
                                     "iota_2266"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_f32, ctx->cuda.module,
                                     "map_transpose_f32"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_f32_low_height,
                                     ctx->cuda.module,
                                     "map_transpose_f32_low_height"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_f32_low_width,
                                     ctx->cuda.module,
                                     "map_transpose_f32_low_width"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_f32_small,
                                     ctx->cuda.module,
                                     "map_transpose_f32_small"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_i32, ctx->cuda.module,
                                     "map_transpose_i32"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_i32_low_height,
                                     ctx->cuda.module,
                                     "map_transpose_i32_low_height"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_i32_low_width,
                                     ctx->cuda.module,
                                     "map_transpose_i32_low_width"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->map_transpose_i32_small,
                                     ctx->cuda.module,
                                     "map_transpose_i32_small"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_2234, ctx->cuda.module,
                                     "replicate_2234"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_2243, ctx->cuda.module,
                                     "replicate_2243"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->replicate_2281, ctx->cuda.module,
                                     "replicate_2281"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->scan_stage1_1689, ctx->cuda.module,
                                     "scan_stage1_1689"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->scan_stage2_1689, ctx->cuda.module,
                                     "scan_stage2_1689"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->scan_stage3_1689, ctx->cuda.module,
                                     "scan_stage3_1689"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1491, ctx->cuda.module,
                                     "segmap_1491"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1587, ctx->cuda.module,
                                     "segmap_1587"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1622, ctx->cuda.module,
                                     "segmap_1622"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1656, ctx->cuda.module,
                                     "segmap_1656"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1702, ctx->cuda.module,
                                     "segmap_1702"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1744, ctx->cuda.module,
                                     "segmap_1744"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1770, ctx->cuda.module,
                                     "segmap_1770"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1786, ctx->cuda.module,
                                     "segmap_1786"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1801, ctx->cuda.module,
                                     "segmap_1801"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segmap_1823, ctx->cuda.module,
                                     "segmap_1823"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segred_large_1372, ctx->cuda.module,
                                     "segred_large_1372"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segred_large_1391, ctx->cuda.module,
                                     "segred_large_1391"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segred_small_1372, ctx->cuda.module,
                                     "segred_small_1372"));
    CUDA_SUCCEED(cuModuleGetFunction(&ctx->segred_small_1391, ctx->cuda.module,
                                     "segred_small_1391"));
    {
        ctx->counter_mem_2146.references = NULL;
        ctx->counter_mem_2146.size = 0;
        CUDA_SUCCEED(cuMemAlloc(&ctx->counter_mem_2146.mem, (10240 >
                                                             0 ? 10240 : 1) *
                                sizeof(int32_t)));
        if (10240 > 0)
            CUDA_SUCCEED(cuMemcpyHtoD(ctx->counter_mem_2146.mem,
                                      counter_mem_realtype_2435, 10240 *
                                      sizeof(int32_t)));
    }
    {
        ctx->counter_mem_2198.references = NULL;
        ctx->counter_mem_2198.size = 0;
        CUDA_SUCCEED(cuMemAlloc(&ctx->counter_mem_2198.mem, (10240 >
                                                             0 ? 10240 : 1) *
                                sizeof(int32_t)));
        if (10240 > 0)
            CUDA_SUCCEED(cuMemcpyHtoD(ctx->counter_mem_2198.mem,
                                      counter_mem_realtype_2454, 10240 *
                                      sizeof(int32_t)));
    }
    ctx->sizes.mainzigroup_sizze_2228 = cfg->sizes[0];
    ctx->sizes.mainzigroup_sizze_2237 = cfg->sizes[1];
    ctx->sizes.mainzigroup_sizze_2246 = cfg->sizes[2];
    ctx->sizes.mainzigroup_sizze_2269 = cfg->sizes[3];
    ctx->sizes.mainzigroup_sizze_2284 = cfg->sizes[4];
    ctx->sizes.mainzisegmap_group_sizze_1494 = cfg->sizes[5];
    ctx->sizes.mainzisegmap_group_sizze_1592 = cfg->sizes[6];
    ctx->sizes.mainzisegmap_group_sizze_1627 = cfg->sizes[7];
    ctx->sizes.mainzisegmap_group_sizze_1659 = cfg->sizes[8];
    ctx->sizes.mainzisegmap_group_sizze_1707 = cfg->sizes[9];
    ctx->sizes.mainzisegmap_group_sizze_1749 = cfg->sizes[10];
    ctx->sizes.mainzisegmap_group_sizze_1773 = cfg->sizes[11];
    ctx->sizes.mainzisegmap_group_sizze_1789 = cfg->sizes[12];
    ctx->sizes.mainzisegmap_group_sizze_1806 = cfg->sizes[13];
    ctx->sizes.mainzisegmap_group_sizze_1826 = cfg->sizes[14];
    ctx->sizes.mainzisegmap_num_groups_1661 = cfg->sizes[15];
    ctx->sizes.mainzisegred_group_sizze_1366 = cfg->sizes[16];
    ctx->sizes.mainzisegred_group_sizze_1385 = cfg->sizes[17];
    ctx->sizes.mainzisegred_num_groups_1368 = cfg->sizes[18];
    ctx->sizes.mainzisegred_num_groups_1387 = cfg->sizes[19];
    ctx->sizes.mainzisegscan_group_sizze_1683 = cfg->sizes[20];
    ctx->sizes.mainzisegscan_num_groups_1685 = cfg->sizes[21];
    init_constants(ctx);
    // Clear the free list of any deallocations that occurred while initialising constants.
    CUDA_SUCCEED(cuda_free_all(&ctx->cuda));
    futhark_context_sync(ctx);
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_constants(ctx);
    cuda_cleanup(&ctx->cuda);
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    CUDA_SUCCEED(cuCtxSynchronize());
    if (ctx->failure_is_an_option) {
        int32_t failure_idx;
        
        CUDA_SUCCEED(cuMemcpyDtoH(&failure_idx, ctx->global_failure,
                                  sizeof(int32_t)));
        ctx->failure_is_an_option = 0;
        if (failure_idx >= 0) {
            int32_t args[4 + 1];
            
            CUDA_SUCCEED(cuMemcpyDtoH(&args, ctx->global_failure_args,
                                      sizeof(args)));
            switch (failure_idx) {
                
              case 0:
                {
                    ctx->error =
                        msgprintf("division by zero\n-> #0  util.fut:51:9-34\n   #1  buildKDtree.fut:33:30-60\n   #2  buildKDtree.fut:107:44-108:76\n   #3  buildKDtree.fut:104:21-117:47\n   #4  buildKDtree.fut:181:11-50\n   #5  buildKDtree.fut:178:1-182:90\n");
                    break;
                }
                
              case 1:
                {
                    ctx->error =
                        msgprintf("Index [%d] out of bounds for array of shape [%d].\n-> #0  buildKDtree.fut:34:23-43\n   #1  buildKDtree.fut:107:44-108:76\n   #2  buildKDtree.fut:104:21-117:47\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 2:
                {
                    ctx->error =
                        msgprintf("Index [%d] out of bounds for array of shape [%d].\n-> #0  buildKDtree.fut:38:75-91\n   #1  buildKDtree.fut:107:44-108:76\n   #2  buildKDtree.fut:104:21-117:47\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 3:
                {
                    ctx->error =
                        msgprintf("Index [%d] out of bounds for array of shape [%d].\n-> #0  buildKDtree.fut:38:9-39:38\n   #1  buildKDtree.fut:107:44-108:76\n   #2  buildKDtree.fut:104:21-117:47\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 4:
                {
                    ctx->error =
                        msgprintf("Index [%d] out of bounds for array of shape [%d].\n-> #0  buildKDtree.fut:49:30-48\n   #1  buildKDtree.fut:115:44-86\n   #2  buildKDtree.fut:104:21-117:47\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 5:
                {
                    ctx->error =
                        msgprintf("Index [%d, %d] out of bounds for array of shape [%d][%d].\n-> #0  buildKDtree.fut:123:58-73\n   #1  buildKDtree.fut:123:45-124:61\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  buildKDtree.fut:122:37-125:59\n   #5  buildKDtree.fut:181:11-50\n   #6  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1], args[2], args[3]);
                    break;
                }
                
              case 6:
                {
                    ctx->error =
                        msgprintf("Index [%d] out of bounds for array of shape [%d].\n-> #0  buildKDtree.fut:148:54-69\n   #1  buildKDtree.fut:148:41-80\n   #2  /prelude/soacs.fut:56:19-23\n   #3  /prelude/soacs.fut:56:3-37\n   #4  buildKDtree.fut:147:31-149:56\n   #5  buildKDtree.fut:181:11-50\n   #6  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1]);
                    break;
                }
                
              case 7:
                {
                    ctx->error =
                        msgprintf("Index [%d, %d] out of bounds for array of shape [%d][%d].\n-> #0  buildKDtree.fut:160:49-62\n   #1  buildKDtree.fut:160:38-72\n   #2  buildKDtree.fut:160:24-81\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n",
                                  args[0], args[1], args[2], args[3]);
                    break;
                }
            }
            return 1;
        }
    }
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    return ctx->error;
}
void futhark_context_pause_profiling(struct futhark_context *ctx)
{
    (void) ctx;
}
void futhark_context_unpause_profiling(struct futhark_context *ctx)
{
    (void) ctx;
}
static int memblock_unref_device(struct futhark_context *ctx,
                                 struct memblock_device *block, const
                                 char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            CUDA_SUCCEED(cuda_free(&ctx->cuda, block->mem, block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc_device(struct futhark_context *ctx,
                                 struct memblock_device *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "space 'device'",
                      ctx->cur_mem_usage_device);
    
    int ret = memblock_unref_device(ctx, block, desc);
    
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    CUDA_SUCCEED(cuda_alloc(&ctx->cuda, size, desc, &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set_device(struct futhark_context *ctx,
                               struct memblock_device *lhs,
                               struct memblock_device *rhs, const
                               char *lhs_desc)
{
    int ret = memblock_unref_device(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1,
                      "Negative allocation of %lld bytes attempted for %s in %s.\n",
                      (long long) size, desc, "default space",
                      ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
static int futrts_main(struct futhark_context *ctx,
                       int32_t *out_scalar_out_2414,
                       int32_t *out_scalar_out_2415,
                       int32_t *out_scalar_out_2416,
                       struct memblock_device *out_mem_p_2417,
                       int32_t *out_out_arrsizze_2418,
                       int32_t *out_out_arrsizze_2419,
                       struct memblock_device *out_mem_p_2420,
                       int32_t *out_out_arrsizze_2421,
                       struct memblock_device *out_mem_p_2422,
                       int32_t *out_out_arrsizze_2423,
                       struct memblock_device *out_mem_p_2424,
                       int32_t *out_out_arrsizze_2425,
                       struct memblock_device *out_mem_p_2426,
                       int32_t *out_out_arrsizze_2427,
                       struct memblock_device input_mem_1905, int32_t m_911,
                       int32_t d_912, int32_t defppl_913);
static int futrts_builtinzhmap_transpose_i32(struct futhark_context *ctx,
                                             struct memblock_device destmem_0,
                                             int32_t destoffset_1,
                                             struct memblock_device srcmem_2,
                                             int32_t srcoffset_3,
                                             int32_t num_arrays_4,
                                             int32_t x_elems_5,
                                             int32_t y_elems_6,
                                             int32_t in_elems_7,
                                             int32_t out_elems_8);
static int futrts_builtinzhreplicate_i32(struct futhark_context *ctx,
                                         struct memblock_device mem_2239,
                                         int32_t num_elems_2240,
                                         int32_t val_2241);
static int futrts_builtinzhreplicate_f32(struct futhark_context *ctx,
                                         struct memblock_device mem_2230,
                                         int32_t num_elems_2231,
                                         float val_2232);
static int futrts_builtinzhmap_transpose_f32(struct futhark_context *ctx,
                                             struct memblock_device destmem_0,
                                             int32_t destoffset_1,
                                             struct memblock_device srcmem_2,
                                             int32_t srcoffset_3,
                                             int32_t num_arrays_4,
                                             int32_t x_elems_5,
                                             int32_t y_elems_6,
                                             int32_t in_elems_7,
                                             int32_t out_elems_8);
int init_constants(struct futhark_context *ctx)
{
    return 0;
}
int free_constants(struct futhark_context *ctx)
{
    return 0;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory || ctx->profiling) {
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->profiling) { }
}
static int futrts_main(struct futhark_context *ctx,
                       int32_t *out_scalar_out_2414,
                       int32_t *out_scalar_out_2415,
                       int32_t *out_scalar_out_2416,
                       struct memblock_device *out_mem_p_2417,
                       int32_t *out_out_arrsizze_2418,
                       int32_t *out_out_arrsizze_2419,
                       struct memblock_device *out_mem_p_2420,
                       int32_t *out_out_arrsizze_2421,
                       struct memblock_device *out_mem_p_2422,
                       int32_t *out_out_arrsizze_2423,
                       struct memblock_device *out_mem_p_2424,
                       int32_t *out_out_arrsizze_2425,
                       struct memblock_device *out_mem_p_2426,
                       int32_t *out_out_arrsizze_2427,
                       struct memblock_device input_mem_1905, int32_t m_911,
                       int32_t d_912, int32_t defppl_913)
{
    int32_t scalar_out_2103;
    int32_t scalar_out_2104;
    int32_t scalar_out_2105;
    struct memblock_device out_mem_2106;
    
    out_mem_2106.references = NULL;
    
    int32_t out_arrsizze_2107;
    int32_t out_arrsizze_2108;
    struct memblock_device out_mem_2109;
    
    out_mem_2109.references = NULL;
    
    int32_t out_arrsizze_2110;
    struct memblock_device out_mem_2111;
    
    out_mem_2111.references = NULL;
    
    int32_t out_arrsizze_2112;
    struct memblock_device out_mem_2113;
    
    out_mem_2113.references = NULL;
    
    int32_t out_arrsizze_2114;
    struct memblock_device out_mem_2115;
    
    out_mem_2115.references = NULL;
    
    int32_t out_arrsizze_2116;
    int32_t x_915 = add32(m_911, defppl_913);
    int32_t x_916 = sub32(x_915, 1);
    bool zzero_917 = defppl_913 == 0;
    bool nonzzero_918 = !zzero_917;
    bool nonzzero_cert_919;
    
    if (!nonzzero_918) {
        ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                               "division by zero",
                               "-> #0  buildKDtree.fut:18:26-50\n   #1  buildKDtree.fut:179:46-70\n   #2  buildKDtree.fut:178:1-182:90\n");
        if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
            return 1;
        return 1;
    }
    
    int32_t def_num_leaves_920 = sdiv32(x_916, defppl_913);
    bool cond_921 = sle32(def_num_leaves_920, 1);
    int32_t res_922;
    
    if (cond_921) {
        res_922 = 0;
    } else {
        bool loop_cond_923 = slt32(1, def_num_leaves_920);
        bool res_924;
        int32_t res_925;
        int32_t res_926;
        bool loop_while_927;
        int32_t q_928;
        int32_t r_929;
        
        loop_while_927 = loop_cond_923;
        q_928 = def_num_leaves_920;
        r_929 = 0;
        while (loop_while_927) {
            int32_t loopres_930 = ashr32(q_928, 1);
            int32_t loopres_931 = add32(1, r_929);
            bool loop_cond_932 = slt32(1, loopres_930);
            bool loop_while_tmp_2117 = loop_cond_932;
            int32_t q_tmp_2118 = loopres_930;
            int32_t r_tmp_2119;
            
            r_tmp_2119 = loopres_931;
            loop_while_927 = loop_while_tmp_2117;
            q_928 = q_tmp_2118;
            r_929 = r_tmp_2119;
        }
        res_924 = loop_while_927;
        res_925 = q_928;
        res_926 = r_929;
        
        int32_t y_933 = 1 << res_926;
        int32_t err_down_934 = sub32(def_num_leaves_920, y_933);
        int32_t y_935 = add32(1, res_926);
        int32_t x_936 = 1 << y_935;
        int32_t err_upwd_937 = sub32(x_936, def_num_leaves_920);
        bool cond_938 = sle32(err_down_934, err_upwd_937);
        int32_t res_939;
        
        if (cond_938) {
            res_939 = res_926;
        } else {
            res_939 = y_935;
        }
        res_922 = res_939;
    }
    
    bool cond_940 = sle32(res_922, 0);
    int32_t res_941;
    int32_t res_942;
    int32_t res_943;
    
    if (cond_940) {
        res_941 = -1;
        res_942 = 0;
        res_943 = m_911;
    } else {
        int32_t h_944 = sub32(res_922, 1);
        int32_t num_leaves_946 = 1 << res_922;
        int32_t x_947 = add32(m_911, num_leaves_946);
        int32_t x_948 = sub32(x_947, 1);
        bool zzero_949 = num_leaves_946 == 0;
        bool nonzzero_950 = !zzero_949;
        bool nonzzero_cert_951;
        
        if (!nonzzero_950) {
            ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                                   "division by zero",
                                   "-> #0  buildKDtree.fut:23:20-52\n   #1  buildKDtree.fut:179:46-70\n   #2  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int32_t ppl_952 = sdiv32(x_948, num_leaves_946);
        int32_t res_953 = sub32(num_leaves_946, 1);
        int32_t res_954 = mul32(num_leaves_946, ppl_952);
        
        res_941 = h_944;
        res_942 = res_953;
        res_943 = res_954;
    }
    
    int64_t m_1362 = sext_i32_i64(m_911);
    int64_t d_1363 = sext_i32_i64(d_912);
    int64_t nest_sizze_1365 = mul64(m_1362, d_1363);
    int32_t segred_group_sizze_1367;
    
    segred_group_sizze_1367 = ctx->sizes.mainzisegred_group_sizze_1366;
    
    int32_t num_groups_1369;
    int32_t max_num_groups_2120;
    
    max_num_groups_2120 = ctx->sizes.mainzisegred_num_groups_1368;
    num_groups_1369 = sext_i64_i32(smax64(1,
                                          smin64(squot64(sub64(add64(nest_sizze_1365,
                                                                     sext_i32_i64(segred_group_sizze_1367)),
                                                               1),
                                                         sext_i32_i64(segred_group_sizze_1367)),
                                                 sext_i32_i64(max_num_groups_2120))));
    
    int32_t convop_x_1907 = mul32(m_911, d_912);
    int64_t binop_x_1908 = sext_i32_i64(convop_x_1907);
    int64_t bytes_1906 = mul64(4, binop_x_1908);
    struct memblock_device mem_1909;
    
    mem_1909.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1909, bytes_1906, "mem_1909"))
        return 1;
    if (futrts_builtinzhmap_transpose_f32(ctx, mem_1909, 0, input_mem_1905, 0,
                                          1, d_912, m_911, mul32(m_911, d_912),
                                          mul32(m_911, d_912)) != 0)
        return 1;
    
    int64_t bytes_1911 = mul64(4, d_1363);
    struct memblock_device mem_1913;
    
    mem_1913.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1913, bytes_1911, "mem_1913"))
        return 1;
    if (slt32(mul32(m_911, 2), segred_group_sizze_1367)) {
        int32_t segment_sizze_nonzzero_2121 = smax32(1, m_911);
        int32_t num_threads_2122 = mul32(num_groups_1369,
                                         segred_group_sizze_1367);
        
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegRed-small");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_segments", (long long) d_912,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segment_size", (long long) m_911,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segments_per_group",
                    (long long) squot32(segred_group_sizze_1367,
                                        segment_sizze_nonzzero_2121), '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "required_groups",
                    (long long) squot32(sub32(add32(d_912,
                                                    squot32(segred_group_sizze_1367,
                                                            segment_sizze_nonzzero_2121)),
                                              1),
                                        squot32(segred_group_sizze_1367,
                                                segment_sizze_nonzzero_2121)),
                    '\n');
        
        unsigned int shared_sizze_2431 = mul32((int32_t) sizeof(float),
                                               segred_group_sizze_1367);
        CUdeviceptr kernel_arg_2433 = mem_1909.mem;
        CUdeviceptr kernel_arg_2434 = mem_1913.mem;
        unsigned int shared_offset_2432 = 0;
        
        if ((((((1 && num_groups_1369 != 0) && 1 != 0) && 1 != 0) &&
              segred_group_sizze_1367 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_1369;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2428[] = {&ctx->global_failure,
                                        &shared_offset_2432, &m_911, &d_912,
                                        &num_groups_1369, &kernel_arg_2433,
                                        &kernel_arg_2434,
                                        &segment_sizze_nonzzero_2121};
            int64_t time_start_2429 = 0, time_end_2430 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (",
                        "segred_small_1372");
                fprintf(stderr, "%d", num_groups_1369);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segred_group_sizze_1367);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2429 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segred_small_1372, grid[0],
                                        grid[1], grid[2],
                                        segred_group_sizze_1367, 1, 1, 0 +
                                        (shared_sizze_2431 + (8 -
                                                              shared_sizze_2431 %
                                                              8) % 8), NULL,
                                        kernel_args_2428, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2430 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                        "segred_small_1372", time_end_2430 - time_start_2429);
            }
        }
    } else {
        int32_t vit_num_groups_2141 = mul32(squot32(sub32(add32(num_groups_1369,
                                                                smax32(1,
                                                                       d_912)),
                                                          1), smax32(1, d_912)),
                                            d_912);
        int32_t num_threads_2142 = mul32(num_groups_1369,
                                         segred_group_sizze_1367);
        int32_t thread_per_segment_2143 =
                mul32(squot32(sub32(add32(num_groups_1369, smax32(1, d_912)),
                                    1), smax32(1, d_912)),
                      segred_group_sizze_1367);
        
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegRed-large");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_segments", (long long) d_912,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segment_size", (long long) m_911,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "virt_num_groups",
                    (long long) vit_num_groups_2141, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_groups",
                    (long long) num_groups_1369, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "group_size",
                    (long long) segred_group_sizze_1367, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_thread",
                    (long long) squot32(sub32(add32(m_911,
                                                    mul32(segred_group_sizze_1367,
                                                          squot32(sub32(add32(num_groups_1369,
                                                                              smax32(1,
                                                                                     d_912)),
                                                                        1),
                                                                  smax32(1,
                                                                         d_912)))),
                                              1), mul32(segred_group_sizze_1367,
                                                        squot32(sub32(add32(num_groups_1369,
                                                                            smax32(1,
                                                                                   d_912)),
                                                                      1),
                                                                smax32(1,
                                                                       d_912)))),
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "groups_per_segment",
                    (long long) squot32(sub32(add32(num_groups_1369, smax32(1,
                                                                            d_912)),
                                              1), smax32(1, d_912)), '\n');
        
        struct memblock_device group_res_arr_mem_2144;
        
        group_res_arr_mem_2144.references = NULL;
        if (memblock_alloc_device(ctx, &group_res_arr_mem_2144,
                                  mul32((int32_t) sizeof(float),
                                        mul32(segred_group_sizze_1367,
                                              vit_num_groups_2141)),
                                  "group_res_arr_mem_2144"))
            return 1;
        
        struct memblock_device counter_mem_2146 = ctx->counter_mem_2146;
        unsigned int shared_sizze_2439 = (int32_t) sizeof(bool);
        unsigned int shared_sizze_2441 = mul32((int32_t) sizeof(float),
                                               segred_group_sizze_1367);
        CUdeviceptr kernel_arg_2443 = mem_1909.mem;
        CUdeviceptr kernel_arg_2444 = mem_1913.mem;
        CUdeviceptr kernel_arg_2445 = group_res_arr_mem_2144.mem;
        CUdeviceptr kernel_arg_2446 = counter_mem_2146.mem;
        unsigned int shared_offset_2440 = 0;
        unsigned int shared_offset_2442 = 0 + (shared_sizze_2439 + (8 -
                                                                    shared_sizze_2439 %
                                                                    8) % 8);
        
        if ((((((1 && num_groups_1369 != 0) && 1 != 0) && 1 != 0) &&
              segred_group_sizze_1367 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_1369;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2436[] = {&ctx->global_failure,
                                        &shared_offset_2440,
                                        &shared_offset_2442, &m_911, &d_912,
                                        &num_groups_1369, &kernel_arg_2443,
                                        &kernel_arg_2444, &vit_num_groups_2141,
                                        &thread_per_segment_2143,
                                        &kernel_arg_2445, &kernel_arg_2446};
            int64_t time_start_2437 = 0, time_end_2438 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (",
                        "segred_large_1372");
                fprintf(stderr, "%d", num_groups_1369);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segred_group_sizze_1367);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2437 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segred_large_1372, grid[0],
                                        grid[1], grid[2],
                                        segred_group_sizze_1367, 1, 1, 0 +
                                        (shared_sizze_2439 + (8 -
                                                              shared_sizze_2439 %
                                                              8) % 8) +
                                        (shared_sizze_2441 + (8 -
                                                              shared_sizze_2441 %
                                                              8) % 8), NULL,
                                        kernel_args_2436, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2438 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                        "segred_large_1372", time_end_2438 - time_start_2437);
            }
        }
        if (memblock_unref_device(ctx, &group_res_arr_mem_2144,
                                  "group_res_arr_mem_2144") != 0)
            return 1;
    }
    
    int32_t segred_group_sizze_1386;
    
    segred_group_sizze_1386 = ctx->sizes.mainzisegred_group_sizze_1385;
    
    int32_t num_groups_1388;
    int32_t max_num_groups_2172;
    
    max_num_groups_2172 = ctx->sizes.mainzisegred_num_groups_1387;
    num_groups_1388 = sext_i64_i32(smax64(1,
                                          smin64(squot64(sub64(add64(nest_sizze_1365,
                                                                     sext_i32_i64(segred_group_sizze_1386)),
                                                               1),
                                                         sext_i32_i64(segred_group_sizze_1386)),
                                                 sext_i32_i64(max_num_groups_2172))));
    
    struct memblock_device mem_1917;
    
    mem_1917.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1917, bytes_1911, "mem_1917"))
        return 1;
    if (slt32(mul32(m_911, 2), segred_group_sizze_1386)) {
        int32_t segment_sizze_nonzzero_2173 = smax32(1, m_911);
        int32_t num_threads_2174 = mul32(num_groups_1388,
                                         segred_group_sizze_1386);
        
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegRed-small");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_segments", (long long) d_912,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segment_size", (long long) m_911,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segments_per_group",
                    (long long) squot32(segred_group_sizze_1386,
                                        segment_sizze_nonzzero_2173), '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "required_groups",
                    (long long) squot32(sub32(add32(d_912,
                                                    squot32(segred_group_sizze_1386,
                                                            segment_sizze_nonzzero_2173)),
                                              1),
                                        squot32(segred_group_sizze_1386,
                                                segment_sizze_nonzzero_2173)),
                    '\n');
        
        unsigned int shared_sizze_2450 = mul32((int32_t) sizeof(float),
                                               segred_group_sizze_1386);
        CUdeviceptr kernel_arg_2452 = mem_1909.mem;
        CUdeviceptr kernel_arg_2453 = mem_1917.mem;
        unsigned int shared_offset_2451 = 0;
        
        if ((((((1 && num_groups_1388 != 0) && 1 != 0) && 1 != 0) &&
              segred_group_sizze_1386 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_1388;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2447[] = {&ctx->global_failure,
                                        &shared_offset_2451, &m_911, &d_912,
                                        &num_groups_1388, &kernel_arg_2452,
                                        &kernel_arg_2453,
                                        &segment_sizze_nonzzero_2173};
            int64_t time_start_2448 = 0, time_end_2449 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (",
                        "segred_small_1391");
                fprintf(stderr, "%d", num_groups_1388);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segred_group_sizze_1386);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2448 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segred_small_1391, grid[0],
                                        grid[1], grid[2],
                                        segred_group_sizze_1386, 1, 1, 0 +
                                        (shared_sizze_2450 + (8 -
                                                              shared_sizze_2450 %
                                                              8) % 8), NULL,
                                        kernel_args_2447, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2449 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                        "segred_small_1391", time_end_2449 - time_start_2448);
            }
        }
    } else {
        int32_t vit_num_groups_2193 = mul32(squot32(sub32(add32(num_groups_1388,
                                                                smax32(1,
                                                                       d_912)),
                                                          1), smax32(1, d_912)),
                                            d_912);
        int32_t num_threads_2194 = mul32(num_groups_1388,
                                         segred_group_sizze_1386);
        int32_t thread_per_segment_2195 =
                mul32(squot32(sub32(add32(num_groups_1388, smax32(1, d_912)),
                                    1), smax32(1, d_912)),
                      segred_group_sizze_1386);
        
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegRed-large");
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_segments", (long long) d_912,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "segment_size", (long long) m_911,
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "virt_num_groups",
                    (long long) vit_num_groups_2193, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "num_groups",
                    (long long) num_groups_1388, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "group_size",
                    (long long) segred_group_sizze_1386, '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "elems_per_thread",
                    (long long) squot32(sub32(add32(m_911,
                                                    mul32(segred_group_sizze_1386,
                                                          squot32(sub32(add32(num_groups_1388,
                                                                              smax32(1,
                                                                                     d_912)),
                                                                        1),
                                                                  smax32(1,
                                                                         d_912)))),
                                              1), mul32(segred_group_sizze_1386,
                                                        squot32(sub32(add32(num_groups_1388,
                                                                            smax32(1,
                                                                                   d_912)),
                                                                      1),
                                                                smax32(1,
                                                                       d_912)))),
                    '\n');
        if (ctx->debugging)
            fprintf(stderr, "%s: %llu%c", "groups_per_segment",
                    (long long) squot32(sub32(add32(num_groups_1388, smax32(1,
                                                                            d_912)),
                                              1), smax32(1, d_912)), '\n');
        
        struct memblock_device group_res_arr_mem_2196;
        
        group_res_arr_mem_2196.references = NULL;
        if (memblock_alloc_device(ctx, &group_res_arr_mem_2196,
                                  mul32((int32_t) sizeof(float),
                                        mul32(segred_group_sizze_1386,
                                              vit_num_groups_2193)),
                                  "group_res_arr_mem_2196"))
            return 1;
        
        struct memblock_device counter_mem_2198 = ctx->counter_mem_2198;
        unsigned int shared_sizze_2458 = (int32_t) sizeof(bool);
        unsigned int shared_sizze_2460 = mul32((int32_t) sizeof(float),
                                               segred_group_sizze_1386);
        CUdeviceptr kernel_arg_2462 = mem_1909.mem;
        CUdeviceptr kernel_arg_2463 = mem_1917.mem;
        CUdeviceptr kernel_arg_2464 = group_res_arr_mem_2196.mem;
        CUdeviceptr kernel_arg_2465 = counter_mem_2198.mem;
        unsigned int shared_offset_2459 = 0;
        unsigned int shared_offset_2461 = 0 + (shared_sizze_2458 + (8 -
                                                                    shared_sizze_2458 %
                                                                    8) % 8);
        
        if ((((((1 && num_groups_1388 != 0) && 1 != 0) && 1 != 0) &&
              segred_group_sizze_1386 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_1388;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2455[] = {&ctx->global_failure,
                                        &shared_offset_2459,
                                        &shared_offset_2461, &m_911, &d_912,
                                        &num_groups_1388, &kernel_arg_2462,
                                        &kernel_arg_2463, &vit_num_groups_2193,
                                        &thread_per_segment_2195,
                                        &kernel_arg_2464, &kernel_arg_2465};
            int64_t time_start_2456 = 0, time_end_2457 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (",
                        "segred_large_1391");
                fprintf(stderr, "%d", num_groups_1388);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segred_group_sizze_1386);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2456 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segred_large_1391, grid[0],
                                        grid[1], grid[2],
                                        segred_group_sizze_1386, 1, 1, 0 +
                                        (shared_sizze_2458 + (8 -
                                                              shared_sizze_2458 %
                                                              8) % 8) +
                                        (shared_sizze_2460 + (8 -
                                                              shared_sizze_2460 %
                                                              8) % 8), NULL,
                                        kernel_args_2455, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2457 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                        "segred_large_1391", time_end_2457 - time_start_2456);
            }
        }
        if (memblock_unref_device(ctx, &group_res_arr_mem_2196,
                                  "group_res_arr_mem_2196") != 0)
            return 1;
    }
    if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
        return 1;
    
    int32_t conc_tmp_972 = add32(d_912, d_912);
    int64_t binop_x_1919 = sext_i32_i64(conc_tmp_972);
    int64_t bytes_1918 = mul64(4, binop_x_1919);
    struct memblock_device mem_1920;
    
    mem_1920.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1920, bytes_1918, "mem_1920"))
        return 1;
    
    int32_t tmp_offs_2224 = 0;
    
    CUDA_SUCCEED(cuMemcpy(mem_1920.mem + mul32(tmp_offs_2224, 4), mem_1913.mem +
                          0, mul64(sext_i32_i64(d_912),
                                   (int32_t) sizeof(float))));
    tmp_offs_2224 += d_912;
    CUDA_SUCCEED(cuMemcpy(mem_1920.mem + mul32(tmp_offs_2224, 4), mem_1917.mem +
                          0, mul64(sext_i32_i64(d_912),
                                   (int32_t) sizeof(float))));
    tmp_offs_2224 += d_912;
    if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
        return 1;
    
    bool bounds_invalid_upwards_981 = slt32(res_943, 0);
    bool valid_982 = !bounds_invalid_upwards_981;
    bool range_valid_c_983;
    
    if (!valid_982) {
        ctx->error = msgprintf("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", res_943,
                               " is invalid.",
                               "-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/array.fut:60:3-12\n   #2  buildKDtree.fut:81:23-29\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n");
        if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_1922 = sext_i32_i64(res_943);
    int64_t bytes_1921 = mul64(4, binop_x_1922);
    struct memblock_device mem_1923;
    
    mem_1923.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1923, bytes_1921, "mem_1923"))
        return 1;
    
    int32_t group_sizze_2228;
    
    group_sizze_2228 = ctx->sizes.mainzigroup_sizze_2228;
    
    int32_t num_groups_2229;
    
    num_groups_2229 = squot32(sub32(add32(res_943,
                                          sext_i32_i32(group_sizze_2228)), 1),
                              sext_i32_i32(group_sizze_2228));
    
    CUdeviceptr kernel_arg_2469 = mem_1923.mem;
    
    if ((((((1 && num_groups_2229 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_2228 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_2229;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_2466[] = {&res_943, &kernel_arg_2469};
        int64_t time_start_2467 = 0, time_end_2468 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "iota_2225");
            fprintf(stderr, "%d", num_groups_2229);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_2228);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_2467 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->iota_2225, grid[0], grid[1], grid[2],
                                    group_sizze_2228, 1, 1, 0, NULL,
                                    kernel_args_2466, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_2468 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "iota_2225",
                    time_end_2468 - time_start_2467);
        }
    }
    
    bool bounds_invalid_upwards_985 = slt32(res_942, 0);
    bool valid_986 = !bounds_invalid_upwards_985;
    bool range_valid_c_987;
    
    if (!valid_986) {
        ctx->error = msgprintf("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n%s",
                               "Range ", 0, "..", 1, "..<", res_942,
                               " is invalid.",
                               "-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/math.fut:454:53-58\n   #2  /prelude/array.fut:70:3-19\n   #3  buildKDtree.fut:83:28-45\n   #4  buildKDtree.fut:181:11-50\n   #5  buildKDtree.fut:178:1-182:90\n");
        if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
            return 1;
        if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_1925 = sext_i32_i64(res_942);
    int64_t bytes_1924 = mul64(4, binop_x_1925);
    struct memblock_device mem_1926;
    
    mem_1926.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1926, bytes_1924, "mem_1926"))
        return 1;
    if (futrts_builtinzhreplicate_f32(ctx, mem_1926, res_942, 0.0F) != 0)
        return 1;
    
    struct memblock_device mem_1929;
    
    mem_1929.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1929, bytes_1924, "mem_1929"))
        return 1;
    if (futrts_builtinzhreplicate_i32(ctx, mem_1929, res_942, -1) != 0)
        return 1;
    
    struct memblock_device mem_1932;
    
    mem_1932.references = NULL;
    if (memblock_alloc_device(ctx, &mem_1932, bytes_1924, "mem_1932"))
        return 1;
    if (futrts_builtinzhreplicate_i32(ctx, mem_1932, res_942, -1) != 0)
        return 1;
    
    int32_t upper_bound_991 = add32(1, res_941);
    int32_t segmap_group_sizze_1495;
    
    segmap_group_sizze_1495 = ctx->sizes.mainzisegmap_group_sizze_1494;
    
    int64_t segmap_group_sizze_1496 = sext_i32_i64(segmap_group_sizze_1495);
    int64_t y_1497 = sub64(segmap_group_sizze_1496, 1);
    int32_t segmap_group_sizze_1774;
    
    segmap_group_sizze_1774 = ctx->sizes.mainzisegmap_group_sizze_1773;
    
    int64_t segmap_group_sizze_1775 = sext_i32_i64(segmap_group_sizze_1774);
    int64_t y_1776 = sub64(segmap_group_sizze_1775, 1);
    int32_t segmap_group_sizze_1750;
    
    segmap_group_sizze_1750 = ctx->sizes.mainzisegmap_group_sizze_1749;
    
    int64_t segmap_group_sizze_1751 = sext_i32_i64(segmap_group_sizze_1750);
    int64_t y_1752 = sub64(segmap_group_sizze_1751, 1);
    int32_t segmap_group_sizze_1708;
    
    segmap_group_sizze_1708 = ctx->sizes.mainzisegmap_group_sizze_1707;
    
    int64_t segmap_group_sizze_1709 = sext_i32_i64(segmap_group_sizze_1708);
    int64_t y_1710 = sub64(segmap_group_sizze_1709, 1);
    int32_t segscan_group_sizze_1684;
    
    segscan_group_sizze_1684 = ctx->sizes.mainzisegscan_group_sizze_1683;
    
    int32_t segmap_group_sizze_1660;
    
    segmap_group_sizze_1660 = ctx->sizes.mainzisegmap_group_sizze_1659;
    
    int32_t segmap_group_sizze_1628;
    
    segmap_group_sizze_1628 = ctx->sizes.mainzisegmap_group_sizze_1627;
    
    int64_t segmap_group_sizze_1629 = sext_i32_i64(segmap_group_sizze_1628);
    int64_t y_1630 = sub64(segmap_group_sizze_1629, 1);
    int32_t segmap_group_sizze_1593;
    
    segmap_group_sizze_1593 = ctx->sizes.mainzisegmap_group_sizze_1592;
    
    int64_t segmap_group_sizze_1594 = sext_i32_i64(segmap_group_sizze_1593);
    int64_t y_1595 = sub64(segmap_group_sizze_1594, 1);
    int32_t segmap_group_sizze_1790;
    
    segmap_group_sizze_1790 = ctx->sizes.mainzisegmap_group_sizze_1789;
    
    int64_t segmap_group_sizze_1791 = sext_i32_i64(segmap_group_sizze_1790);
    int64_t y_1792 = sub64(segmap_group_sizze_1791, 1);
    struct memblock_device res_mem_2045;
    
    res_mem_2045.references = NULL;
    
    struct memblock_device indir_mem_1933;
    
    indir_mem_1933.references = NULL;
    if (memblock_set_device(ctx, &indir_mem_1933, &mem_1923, "mem_1923") != 0)
        return 1;
    for (int32_t lev_1008 = 0; lev_1008 < upper_bound_991; lev_1008++) {
        int32_t nodes_this_lvl_1009 = 1 << lev_1008;
        bool zzero_1010 = nodes_this_lvl_1009 == 0;
        bool nonzzero_1011 = !zzero_1010;
        bool nonzzero_cert_1012;
        
        if (!nonzzero_1011) {
            ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                                   "division by zero",
                                   "-> #0  buildKDtree.fut:97:42-60\n   #1  buildKDtree.fut:181:11-50\n   #2  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int32_t pts_per_node_at_lev_1013 = sdiv32(res_943, nodes_this_lvl_1009);
        int32_t x_1014 = mul32(nodes_this_lvl_1009, pts_per_node_at_lev_1013);
        bool assert_arg_1015 = x_1014 == res_943;
        bool dim_ok_1016;
        
        if (!assert_arg_1015) {
            ctx->error = msgprintf("Error: %s\n\nBacktrace:\n%s",
                                   "new shape has different number of elements than old shape",
                                   "-> #0  /prelude/array.fut:95:3-33\n   #1  buildKDtree.fut:98:30-79\n   #2  buildKDtree.fut:181:11-50\n   #3  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        bool bounds_invalid_upwards_1018 = slt32(nodes_this_lvl_1009, 0);
        bool valid_1019 = !bounds_invalid_upwards_1018;
        bool range_valid_c_1020;
        
        if (!valid_1019) {
            ctx->error = msgprintf("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n%s",
                                   "Range ", 0, "..", 1, "..<",
                                   nodes_this_lvl_1009, " is invalid.",
                                   "-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/array.fut:60:3-12\n   #2  buildKDtree.fut:117:28-46\n   #3  buildKDtree.fut:181:11-50\n   #4  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int64_t nodes_this_lvl_1492 = sext_i32_i64(nodes_this_lvl_1009);
        int64_t x_1498 = add64(nodes_this_lvl_1492, y_1497);
        int64_t segmap_usable_groups_64_1500 = squot64(x_1498,
                                                       segmap_group_sizze_1496);
        int32_t segmap_usable_groups_1501 =
                sext_i64_i32(segmap_usable_groups_64_1500);
        int64_t bytes_1941 = mul64(4, nodes_this_lvl_1492);
        struct memblock_device mem_1943;
        
        mem_1943.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1943, bytes_1941, "mem_1943"))
            return 1;
        
        struct memblock_device mem_1946;
        
        mem_1946.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1946, bytes_1941, "mem_1946"))
            return 1;
        
        int32_t num_threads_2075 = mul32(segmap_group_sizze_1495,
                                         segmap_usable_groups_1501);
        int64_t num_threads64_2076 = sext_i32_i64(num_threads_2075);
        int64_t total_sizze_2077 = mul64(bytes_1918, num_threads64_2076);
        struct memblock_device mem_1940;
        
        mem_1940.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1940, total_sizze_2077, "mem_1940"))
            return 1;
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegMap");
        
        CUdeviceptr kernel_arg_2473 = mem_1920.mem;
        CUdeviceptr kernel_arg_2474 = mem_1926.mem;
        CUdeviceptr kernel_arg_2475 = mem_1929.mem;
        CUdeviceptr kernel_arg_2476 = mem_1940.mem;
        CUdeviceptr kernel_arg_2477 = mem_1943.mem;
        CUdeviceptr kernel_arg_2478 = mem_1946.mem;
        
        if ((((((1 && segmap_usable_groups_1501 != 0) && 1 != 0) && 1 != 0) &&
              segmap_group_sizze_1495 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = segmap_usable_groups_1501;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2470[] = {&ctx->global_failure,
                                        &ctx->failure_is_an_option,
                                        &ctx->global_failure_args, &d_912,
                                        &res_942, &conc_tmp_972, &lev_1008,
                                        &nodes_this_lvl_1009,
                                        &segmap_usable_groups_1501,
                                        &kernel_arg_2473, &kernel_arg_2474,
                                        &kernel_arg_2475, &kernel_arg_2476,
                                        &kernel_arg_2477, &kernel_arg_2478};
            int64_t time_start_2471 = 0, time_end_2472 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "segmap_1491");
                fprintf(stderr, "%d", segmap_usable_groups_1501);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segmap_group_sizze_1495);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2471 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1491, grid[0], grid[1],
                                        grid[2], segmap_group_sizze_1495, 1, 1,
                                        0, NULL, kernel_args_2470, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2472 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1491",
                        time_end_2472 - time_start_2471);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
            return 1;
        
        bool bounds_invalid_upwards_1112 = slt32(pts_per_node_at_lev_1013, 0);
        bool valid_1113 = !bounds_invalid_upwards_1112;
        bool range_valid_c_1114;
        
        if (!valid_1113) {
            ctx->error = msgprintf("Error: %s%d%s%d%s%d%s\n\nBacktrace:\n%s",
                                   "Range ", 0, "..", 1, "..<",
                                   pts_per_node_at_lev_1013, " is invalid.",
                                   "-> #0  /prelude/math.fut:453:23-30\n   #1  /prelude/array.fut:60:3-12\n   #2  lib/github.com/diku-dk/sorts/radix_sort.fut:49:11-16\n   #3  /prelude/functional.fut:9:42-44\n   #4  lib/github.com/diku-dk/sorts/radix_sort.fut:52:3-53:17\n   #5  lib/github.com/diku-dk/sorts/radix_sort.fut:103:3-57\n   #6  /prelude/functional.fut:9:42-44\n   #7  buildKDtree.fut:128:21-129:91\n   #8  buildKDtree.fut:181:11-50\n   #9  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
                return 1;
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int64_t binop_x_1948 = sext_i32_i64(pts_per_node_at_lev_1013);
        int64_t bytes_1947 = mul64(4, binop_x_1948);
        struct memblock_device mem_1949;
        
        mem_1949.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1949, bytes_1947, "mem_1949"))
            return 1;
        
        int32_t group_sizze_2269;
        
        group_sizze_2269 = ctx->sizes.mainzigroup_sizze_2269;
        
        int32_t num_groups_2270;
        
        num_groups_2270 = squot32(sub32(add32(pts_per_node_at_lev_1013,
                                              sext_i32_i32(group_sizze_2269)),
                                        1), sext_i32_i32(group_sizze_2269));
        
        CUdeviceptr kernel_arg_2482 = mem_1949.mem;
        
        if ((((((1 && num_groups_2270 != 0) && 1 != 0) && 1 != 0) &&
              group_sizze_2269 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_2270;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2479[] = {&pts_per_node_at_lev_1013,
                                        &kernel_arg_2482};
            int64_t time_start_2480 = 0, time_end_2481 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "iota_2266");
                fprintf(stderr, "%d", num_groups_2270);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", group_sizze_2269);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2480 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->iota_2266, grid[0], grid[1],
                                        grid[2], group_sizze_2269, 1, 1, 0,
                                        NULL, kernel_args_2479, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2481 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "iota_2266",
                        time_end_2481 - time_start_2480);
            }
        }
        
        int32_t mi_1208 = sdiv32(pts_per_node_at_lev_1013, 2);
        bool x_1209 = sle32(0, mi_1208);
        bool y_1210 = slt32(mi_1208, pts_per_node_at_lev_1013);
        bool bounds_check_1211 = x_1209 && y_1210;
        bool index_certs_1212;
        
        if (!bounds_check_1211) {
            ctx->error = msgprintf("Error: %s%d%s%d%s\n\nBacktrace:\n%s",
                                   "Index [", mi_1208,
                                   "] out of bounds for array of shape [",
                                   pts_per_node_at_lev_1013, "].",
                                   "-> #0  buildKDtree.fut:144:45-58\n   #1  buildKDtree.fut:142:31-145:50\n   #2  buildKDtree.fut:181:11-50\n   #3  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &mem_1949, "mem_1949") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
                return 1;
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int32_t i_1213 = sub32(mi_1208, 1);
        bool x_1214 = sle32(0, i_1213);
        bool y_1215 = slt32(i_1213, pts_per_node_at_lev_1013);
        bool bounds_check_1216 = x_1214 && y_1215;
        bool index_certs_1217;
        
        if (!bounds_check_1216) {
            ctx->error = msgprintf("Error: %s%d%s%d%s\n\nBacktrace:\n%s",
                                   "Index [", i_1213,
                                   "] out of bounds for array of shape [",
                                   pts_per_node_at_lev_1013, "].",
                                   "-> #0  buildKDtree.fut:144:62-77\n   #1  buildKDtree.fut:142:31-145:50\n   #2  buildKDtree.fut:181:11-50\n   #3  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &mem_1949, "mem_1949") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
                return 1;
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        int64_t x_1777 = add64(nodes_this_lvl_1492, y_1776);
        int64_t segmap_usable_groups_64_1779 = squot64(x_1777,
                                                       segmap_group_sizze_1775);
        int32_t segmap_usable_groups_1780 =
                sext_i64_i32(segmap_usable_groups_64_1779);
        struct memblock_device mem_1952;
        
        mem_1952.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1952, nodes_this_lvl_1492,
                                  "mem_1952"))
            return 1;
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegMap");
        
        CUdeviceptr kernel_arg_2486 = mem_1943.mem;
        CUdeviceptr kernel_arg_2487 = mem_1952.mem;
        
        if ((((((1 && segmap_usable_groups_1780 != 0) && 1 != 0) && 1 != 0) &&
              segmap_group_sizze_1774 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = segmap_usable_groups_1780;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2483[] = {&ctx->global_failure, &d_912,
                                        &nodes_this_lvl_1009, &kernel_arg_2486,
                                        &kernel_arg_2487};
            int64_t time_start_2484 = 0, time_end_2485 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "segmap_1770");
                fprintf(stderr, "%d", segmap_usable_groups_1780);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segmap_group_sizze_1774);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2484 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1770, grid[0], grid[1],
                                        grid[2], segmap_group_sizze_1774, 1, 1,
                                        0, NULL, kernel_args_2483, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2485 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1770",
                        time_end_2485 - time_start_2484);
            }
        }
        
        int64_t nest_sizze_1748 = mul64(nodes_this_lvl_1492, binop_x_1948);
        int64_t x_1753 = add64(nest_sizze_1748, y_1752);
        int64_t segmap_usable_groups_64_1755 = squot64(x_1753,
                                                       segmap_group_sizze_1751);
        int32_t segmap_usable_groups_1756 =
                sext_i64_i32(segmap_usable_groups_64_1755);
        int64_t bytes_1954 = mul64(4, nest_sizze_1748);
        struct memblock_device mem_1958;
        
        mem_1958.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1958, bytes_1954, "mem_1958"))
            return 1;
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegMap");
        
        CUdeviceptr kernel_arg_2491 = input_mem_1905.mem;
        CUdeviceptr kernel_arg_2492 = indir_mem_1933.mem;
        CUdeviceptr kernel_arg_2493 = mem_1943.mem;
        CUdeviceptr kernel_arg_2494 = mem_1952.mem;
        CUdeviceptr kernel_arg_2495 = mem_1958.mem;
        
        if ((((((1 && segmap_usable_groups_1756 != 0) && 1 != 0) && 1 != 0) &&
              segmap_group_sizze_1750 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = segmap_usable_groups_1756;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2488[] = {&ctx->global_failure,
                                        &ctx->failure_is_an_option,
                                        &ctx->global_failure_args, &m_911,
                                        &d_912, &res_943, &nodes_this_lvl_1009,
                                        &pts_per_node_at_lev_1013,
                                        &kernel_arg_2491, &kernel_arg_2492,
                                        &kernel_arg_2493, &kernel_arg_2494,
                                        &kernel_arg_2495};
            int64_t time_start_2489 = 0, time_end_2490 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "segmap_1744");
                fprintf(stderr, "%d", segmap_usable_groups_1756);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segmap_group_sizze_1750);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2489 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1744, grid[0], grid[1],
                                        grid[2], segmap_group_sizze_1750, 1, 1,
                                        0, NULL, kernel_args_2488, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2490 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1744",
                        time_end_2490 - time_start_2489);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_unref_device(ctx, &mem_1952, "mem_1952") != 0)
            return 1;
        
        struct memblock_device mem_1963;
        
        mem_1963.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1963, bytes_1954, "mem_1963"))
            return 1;
        
        int32_t group_sizze_2284;
        
        group_sizze_2284 = ctx->sizes.mainzigroup_sizze_2284;
        
        int32_t num_groups_2285;
        
        num_groups_2285 = squot32(sub32(add32(mul32(nodes_this_lvl_1009,
                                                    pts_per_node_at_lev_1013),
                                              sext_i32_i32(group_sizze_2284)),
                                        1), sext_i32_i32(group_sizze_2284));
        
        CUdeviceptr kernel_arg_2499 = mem_1949.mem;
        CUdeviceptr kernel_arg_2500 = mem_1963.mem;
        
        if ((((((1 && num_groups_2285 != 0) && 1 != 0) && 1 != 0) &&
              group_sizze_2284 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = num_groups_2285;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2496[] = {&nodes_this_lvl_1009,
                                        &pts_per_node_at_lev_1013,
                                        &kernel_arg_2499, &kernel_arg_2500};
            int64_t time_start_2497 = 0, time_end_2498 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (",
                        "replicate_2281");
                fprintf(stderr, "%d", num_groups_2285);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", group_sizze_2284);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2497 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_2281, grid[0], grid[1],
                                        grid[2], group_sizze_2284, 1, 1, 0,
                                        NULL, kernel_args_2496, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2498 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_2281",
                        time_end_2498 - time_start_2497);
            }
        }
        if (memblock_unref_device(ctx, &mem_1949, "mem_1949") != 0)
            return 1;
        
        int64_t x_1711 = add64(y_1710, nest_sizze_1748);
        int64_t x_1849 = squot64(x_1711, segmap_group_sizze_1709);
        int32_t segmap_usable_groups_1714 = sext_i64_i32(x_1849);
        int32_t num_groups_1686;
        int32_t max_num_groups_2286;
        
        max_num_groups_2286 = ctx->sizes.mainzisegscan_num_groups_1685;
        num_groups_1686 = sext_i64_i32(smax64(1,
                                              smin64(squot64(sub64(add64(nest_sizze_1748,
                                                                         sext_i32_i64(segscan_group_sizze_1684)),
                                                                   1),
                                                             sext_i32_i64(segscan_group_sizze_1684)),
                                                     sext_i32_i64(max_num_groups_2286))));
        
        int32_t num_groups_1662;
        int32_t max_num_groups_2287;
        
        max_num_groups_2287 = ctx->sizes.mainzisegmap_num_groups_1661;
        num_groups_1662 = sext_i64_i32(smax64(1,
                                              smin64(squot64(sub64(add64(nodes_this_lvl_1492,
                                                                         sext_i32_i64(segmap_group_sizze_1660)),
                                                                   1),
                                                             sext_i32_i64(segmap_group_sizze_1660)),
                                                     sext_i32_i64(max_num_groups_2287))));
        
        int64_t x_1631 = add64(y_1630, nest_sizze_1748);
        int64_t x_1851 = squot64(x_1631, segmap_group_sizze_1629);
        int32_t segmap_usable_groups_1634 = sext_i64_i32(x_1851);
        struct memblock_device mem_1971;
        
        mem_1971.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1971, bytes_1954, "mem_1971"))
            return 1;
        
        struct memblock_device mem_1977;
        
        mem_1977.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1977, bytes_1954, "mem_1977"))
            return 1;
        
        struct memblock_device mem_1982;
        
        mem_1982.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1982, bytes_1954, "mem_1982"))
            return 1;
        
        struct memblock_device mem_1987;
        
        mem_1987.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1987, bytes_1954, "mem_1987"))
            return 1;
        
        int64_t binop_x_1990 = sext_i32_i64(x_1014);
        int64_t bytes_1988 = mul64(4, binop_x_1990);
        struct memblock_device mem_1991;
        
        mem_1991.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1991, bytes_1988, "mem_1991"))
            return 1;
        
        struct memblock_device mem_1995;
        
        mem_1995.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1995, bytes_1988, "mem_1995"))
            return 1;
        
        struct memblock_device mem_1999;
        
        mem_1999.references = NULL;
        if (memblock_alloc_device(ctx, &mem_1999, bytes_1988, "mem_1999"))
            return 1;
        
        struct memblock_device mem_2009;
        
        mem_2009.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2009, bytes_1941, "mem_2009"))
            return 1;
        
        struct memblock_device mem_2013;
        
        mem_2013.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2013, bytes_1988, "mem_2013"))
            return 1;
        
        struct memblock_device mem_2017;
        
        mem_2017.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2017, bytes_1988, "mem_2017"))
            return 1;
        
        int32_t num_threads_2086 = mul32(segmap_group_sizze_1660,
                                         num_groups_1662);
        int64_t num_threads64_2087 = sext_i32_i64(num_threads_2086);
        int64_t total_sizze_2088 = mul64(bytes_1947, num_threads64_2087);
        struct memblock_device mem_2003;
        
        mem_2003.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2003, total_sizze_2088, "mem_2003"))
            return 1;
        
        int64_t total_sizze_2089 = mul64(bytes_1947, num_threads64_2087);
        struct memblock_device mem_2006;
        
        mem_2006.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2006, total_sizze_2089, "mem_2006"))
            return 1;
        
        struct memblock_device res_mem_2031;
        
        res_mem_2031.references = NULL;
        
        struct memblock_device res_r_mem_2032;
        
        res_r_mem_2032.references = NULL;
        
        struct memblock_device xs_expanded_mem_1964;
        
        xs_expanded_mem_1964.references = NULL;
        
        struct memblock_device xs_expanded_mem_1965;
        
        xs_expanded_mem_1965.references = NULL;
        if (memblock_set_device(ctx, &xs_expanded_mem_1964, &mem_1958,
                                "mem_1958") != 0)
            return 1;
        if (memblock_set_device(ctx, &xs_expanded_mem_1965, &mem_1963,
                                "mem_1963") != 0)
            return 1;
        for (int32_t i_1241 = 0; i_1241 < 32; i_1241++) {
            bool res_1740 = i_1241 == 31;
            
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            
            CUdeviceptr kernel_arg_2504 = xs_expanded_mem_1964.mem;
            CUdeviceptr kernel_arg_2505 = mem_1971.mem;
            
            if ((((((1 && segmap_usable_groups_1714 != 0) && 1 != 0) && 1 !=
                   0) && segmap_group_sizze_1708 != 0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = segmap_usable_groups_1714;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2501[] = {&ctx->global_failure,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013, &i_1241,
                                            &res_1740, &kernel_arg_2504,
                                            &kernel_arg_2505};
                int64_t time_start_2502 = 0, time_end_2503 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "segmap_1702");
                    fprintf(stderr, "%d", segmap_usable_groups_1714);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", segmap_group_sizze_1708);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2502 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1702, grid[0], grid[1],
                                            grid[2], segmap_group_sizze_1708, 1,
                                            1, 0, NULL, kernel_args_2501,
                                            NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2503 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1702",
                            time_end_2503 - time_start_2502);
                }
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegScan");
            
            int32_t num_threads_2297 = mul32(num_groups_1686,
                                             segscan_group_sizze_1684);
            unsigned int shared_sizze_2509 = mul32((int32_t) sizeof(int32_t),
                                                   segscan_group_sizze_1684);
            unsigned int shared_sizze_2511 = mul32((int32_t) sizeof(int32_t),
                                                   segscan_group_sizze_1684);
            CUdeviceptr kernel_arg_2513 = mem_1971.mem;
            CUdeviceptr kernel_arg_2514 = mem_1977.mem;
            CUdeviceptr kernel_arg_2515 = mem_1982.mem;
            CUdeviceptr kernel_arg_2516 = mem_1987.mem;
            unsigned int shared_offset_2510 = 0;
            unsigned int shared_offset_2512 = 0 + (shared_sizze_2509 + (8 -
                                                                        shared_sizze_2509 %
                                                                        8) % 8);
            
            if ((((((1 && num_groups_1686 != 0) && 1 != 0) && 1 != 0) &&
                  segscan_group_sizze_1684 != 0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = num_groups_1686;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2506[] = {&ctx->global_failure,
                                            &shared_offset_2510,
                                            &shared_offset_2512,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013,
                                            &kernel_arg_2513, &kernel_arg_2514,
                                            &kernel_arg_2515, &kernel_arg_2516,
                                            &num_threads_2297};
                int64_t time_start_2507 = 0, time_end_2508 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "scan_stage1_1689");
                    fprintf(stderr, "%d", num_groups_1686);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", segscan_group_sizze_1684);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2507 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->scan_stage1_1689, grid[0],
                                            grid[1], grid[2],
                                            segscan_group_sizze_1684, 1, 1, 0 +
                                            (shared_sizze_2509 + (8 -
                                                                  shared_sizze_2509 %
                                                                  8) % 8) +
                                            (shared_sizze_2511 + (8 -
                                                                  shared_sizze_2511 %
                                                                  8) % 8), NULL,
                                            kernel_args_2506, NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2508 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n",
                            "scan_stage1_1689", time_end_2508 -
                            time_start_2507);
                }
            }
            if (ctx->debugging)
                fprintf(stderr, "%s: %llu%c", "elems_per_group",
                        (long long) mul32(segscan_group_sizze_1684,
                                          squot32(sub32(add32(mul32(nodes_this_lvl_1009,
                                                                    pts_per_node_at_lev_1013),
                                                              num_threads_2297),
                                                        1), num_threads_2297)),
                        '\n');
            
            unsigned int shared_sizze_2520 = mul32((int32_t) sizeof(int32_t),
                                                   num_groups_1686);
            unsigned int shared_sizze_2522 = mul32((int32_t) sizeof(int32_t),
                                                   num_groups_1686);
            CUdeviceptr kernel_arg_2524 = mem_1977.mem;
            CUdeviceptr kernel_arg_2525 = mem_1982.mem;
            unsigned int shared_offset_2521 = 0;
            unsigned int shared_offset_2523 = 0 + (shared_sizze_2520 + (8 -
                                                                        shared_sizze_2520 %
                                                                        8) % 8);
            
            if ((((((1 && 1 != 0) && 1 != 0) && 1 != 0) && num_groups_1686 !=
                  0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = 1;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2517[] = {&ctx->global_failure,
                                            &shared_offset_2521,
                                            &shared_offset_2523,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013,
                                            &num_groups_1686, &kernel_arg_2524,
                                            &kernel_arg_2525,
                                            &num_threads_2297};
                int64_t time_start_2518 = 0, time_end_2519 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "scan_stage2_1689");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", num_groups_1686);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2518 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->scan_stage2_1689, grid[0],
                                            grid[1], grid[2], num_groups_1686,
                                            1, 1, 0 + (shared_sizze_2520 + (8 -
                                                                            shared_sizze_2520 %
                                                                            8) %
                                                       8) + (shared_sizze_2522 +
                                                             (8 -
                                                              shared_sizze_2522 %
                                                              8) % 8), NULL,
                                            kernel_args_2517, NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2519 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n",
                            "scan_stage2_1689", time_end_2519 -
                            time_start_2518);
                }
            }
            
            int32_t required_groups_2362 =
                    squot32(sub32(add32(mul32(nodes_this_lvl_1009,
                                              pts_per_node_at_lev_1013),
                                        segscan_group_sizze_1684), 1),
                            segscan_group_sizze_1684);
            CUdeviceptr kernel_arg_2529 = mem_1977.mem;
            CUdeviceptr kernel_arg_2530 = mem_1982.mem;
            
            if ((((((1 && num_groups_1686 != 0) && 1 != 0) && 1 != 0) &&
                  segscan_group_sizze_1684 != 0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = num_groups_1686;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2526[] = {&ctx->global_failure,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013,
                                            &num_groups_1686, &kernel_arg_2529,
                                            &kernel_arg_2530, &num_threads_2297,
                                            &required_groups_2362};
                int64_t time_start_2527 = 0, time_end_2528 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "scan_stage3_1689");
                    fprintf(stderr, "%d", num_groups_1686);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", segscan_group_sizze_1684);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2527 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->scan_stage3_1689, grid[0],
                                            grid[1], grid[2],
                                            segscan_group_sizze_1684, 1, 1, 0,
                                            NULL, kernel_args_2526, NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2528 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n",
                            "scan_stage3_1689", time_end_2528 -
                            time_start_2527);
                }
            }
            if (futrts_builtinzhmap_transpose_f32(ctx, mem_1991, 0,
                                                  xs_expanded_mem_1964, 0, 1,
                                                  pts_per_node_at_lev_1013,
                                                  nodes_this_lvl_1009,
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013),
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013)) !=
                0)
                return 1;
            if (futrts_builtinzhmap_transpose_i32(ctx, mem_1995, 0,
                                                  xs_expanded_mem_1965, 0, 1,
                                                  pts_per_node_at_lev_1013,
                                                  nodes_this_lvl_1009,
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013),
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013)) !=
                0)
                return 1;
            if (futrts_builtinzhmap_transpose_i32(ctx, mem_1999, 0, mem_1987, 0,
                                                  1, pts_per_node_at_lev_1013,
                                                  nodes_this_lvl_1009,
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013),
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013)) !=
                0)
                return 1;
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            
            CUdeviceptr kernel_arg_2534 = mem_1991.mem;
            CUdeviceptr kernel_arg_2535 = mem_1995.mem;
            CUdeviceptr kernel_arg_2536 = mem_1999.mem;
            CUdeviceptr kernel_arg_2537 = mem_2003.mem;
            CUdeviceptr kernel_arg_2538 = mem_2006.mem;
            CUdeviceptr kernel_arg_2539 = mem_2009.mem;
            CUdeviceptr kernel_arg_2540 = mem_2013.mem;
            CUdeviceptr kernel_arg_2541 = mem_2017.mem;
            
            if ((((((1 && num_groups_1662 != 0) && 1 != 0) && 1 != 0) &&
                  segmap_group_sizze_1660 != 0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = num_groups_1662;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2531[] = {&ctx->global_failure,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013,
                                            &num_groups_1662, &kernel_arg_2534,
                                            &kernel_arg_2535, &kernel_arg_2536,
                                            &kernel_arg_2537, &kernel_arg_2538,
                                            &kernel_arg_2539, &kernel_arg_2540,
                                            &kernel_arg_2541};
                int64_t time_start_2532 = 0, time_end_2533 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "segmap_1656");
                    fprintf(stderr, "%d", num_groups_1662);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", segmap_group_sizze_1660);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2532 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1656, grid[0], grid[1],
                                            grid[2], segmap_group_sizze_1660, 1,
                                            1, 0, NULL, kernel_args_2531,
                                            NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2533 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1656",
                            time_end_2533 - time_start_2532);
                }
            }
            if (ctx->debugging)
                fprintf(stderr, "%s\n", "\n# SegMap");
            
            CUdeviceptr kernel_arg_2545 = xs_expanded_mem_1964.mem;
            CUdeviceptr kernel_arg_2546 = xs_expanded_mem_1965.mem;
            CUdeviceptr kernel_arg_2547 = mem_1971.mem;
            CUdeviceptr kernel_arg_2548 = mem_1977.mem;
            CUdeviceptr kernel_arg_2549 = mem_1982.mem;
            CUdeviceptr kernel_arg_2550 = mem_1987.mem;
            CUdeviceptr kernel_arg_2551 = mem_2009.mem;
            CUdeviceptr kernel_arg_2552 = mem_2013.mem;
            CUdeviceptr kernel_arg_2553 = mem_2017.mem;
            
            if ((((((1 && segmap_usable_groups_1634 != 0) && 1 != 0) && 1 !=
                   0) && segmap_group_sizze_1628 != 0) && 1 != 0) && 1 != 0) {
                int perm[3] = {0, 1, 2};
                
                if (1 > 1 << 16) {
                    perm[1] = perm[0];
                    perm[0] = 1;
                }
                if (1 > 1 << 16) {
                    perm[2] = perm[0];
                    perm[0] = 2;
                }
                
                size_t grid[3];
                
                grid[perm[0]] = segmap_usable_groups_1634;
                grid[perm[1]] = 1;
                grid[perm[2]] = 1;
                
                void *kernel_args_2542[] = {&ctx->global_failure,
                                            &nodes_this_lvl_1009,
                                            &pts_per_node_at_lev_1013,
                                            &kernel_arg_2545, &kernel_arg_2546,
                                            &kernel_arg_2547, &kernel_arg_2548,
                                            &kernel_arg_2549, &kernel_arg_2550,
                                            &kernel_arg_2551, &kernel_arg_2552,
                                            &kernel_arg_2553};
                int64_t time_start_2543 = 0, time_end_2544 = 0;
                
                if (ctx->debugging) {
                    fprintf(stderr, "Launching %s with grid size (",
                            "segmap_1622");
                    fprintf(stderr, "%d", segmap_usable_groups_1634);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ") and block size (");
                    fprintf(stderr, "%d", segmap_group_sizze_1628);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ", ");
                    fprintf(stderr, "%d", 1);
                    fprintf(stderr, ").\n");
                    time_start_2543 = get_wall_time();
                }
                CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1622, grid[0], grid[1],
                                            grid[2], segmap_group_sizze_1628, 1,
                                            1, 0, NULL, kernel_args_2542,
                                            NULL));
                if (ctx->debugging) {
                    CUDA_SUCCEED(cuCtxSynchronize());
                    time_end_2544 = get_wall_time();
                    fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1622",
                            time_end_2544 - time_start_2543);
                }
            }
            
            struct memblock_device mem_2023;
            
            mem_2023.references = NULL;
            if (memblock_alloc_device(ctx, &mem_2023, bytes_1954, "mem_2023"))
                return 1;
            if (futrts_builtinzhmap_transpose_f32(ctx, mem_2023, 0, mem_2017, 0,
                                                  1, nodes_this_lvl_1009,
                                                  pts_per_node_at_lev_1013,
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013),
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013)) !=
                0)
                return 1;
            
            struct memblock_device mem_2029;
            
            mem_2029.references = NULL;
            if (memblock_alloc_device(ctx, &mem_2029, bytes_1954, "mem_2029"))
                return 1;
            if (futrts_builtinzhmap_transpose_i32(ctx, mem_2029, 0, mem_2013, 0,
                                                  1, nodes_this_lvl_1009,
                                                  pts_per_node_at_lev_1013,
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013),
                                                  mul32(nodes_this_lvl_1009,
                                                        pts_per_node_at_lev_1013)) !=
                0)
                return 1;
            
            struct memblock_device xs_expanded_mem_tmp_2288;
            
            xs_expanded_mem_tmp_2288.references = NULL;
            if (memblock_set_device(ctx, &xs_expanded_mem_tmp_2288, &mem_2023,
                                    "mem_2023") != 0)
                return 1;
            
            struct memblock_device xs_expanded_mem_tmp_2289;
            
            xs_expanded_mem_tmp_2289.references = NULL;
            if (memblock_set_device(ctx, &xs_expanded_mem_tmp_2289, &mem_2029,
                                    "mem_2029") != 0)
                return 1;
            if (memblock_set_device(ctx, &xs_expanded_mem_1964,
                                    &xs_expanded_mem_tmp_2288,
                                    "xs_expanded_mem_tmp_2288") != 0)
                return 1;
            if (memblock_set_device(ctx, &xs_expanded_mem_1965,
                                    &xs_expanded_mem_tmp_2289,
                                    "xs_expanded_mem_tmp_2289") != 0)
                return 1;
            if (memblock_unref_device(ctx, &xs_expanded_mem_tmp_2289,
                                      "xs_expanded_mem_tmp_2289") != 0)
                return 1;
            if (memblock_unref_device(ctx, &xs_expanded_mem_tmp_2288,
                                      "xs_expanded_mem_tmp_2288") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2029, "mem_2029") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2023, "mem_2023") != 0)
                return 1;
        }
        if (memblock_set_device(ctx, &res_mem_2031, &xs_expanded_mem_1964,
                                "xs_expanded_mem_1964") != 0)
            return 1;
        if (memblock_set_device(ctx, &res_r_mem_2032, &xs_expanded_mem_1965,
                                "xs_expanded_mem_1965") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1963, "mem_1963") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1971, "mem_1971") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1977, "mem_1977") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1982, "mem_1982") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1987, "mem_1987") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1991, "mem_1991") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1995, "mem_1995") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1999, "mem_1999") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2003, "mem_2003") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2006, "mem_2006") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2009, "mem_2009") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2013, "mem_2013") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2017, "mem_2017") != 0)
            return 1;
        
        int64_t x_1596 = add64(y_1595, nest_sizze_1748);
        int64_t segmap_usable_groups_64_1598 = squot64(x_1596,
                                                       segmap_group_sizze_1594);
        int32_t segmap_usable_groups_1599 =
                sext_i64_i32(segmap_usable_groups_64_1598);
        struct memblock_device mem_2038;
        
        mem_2038.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2038, bytes_1954, "mem_2038"))
            return 1;
        
        struct memblock_device mem_2043;
        
        mem_2043.references = NULL;
        if (memblock_alloc_device(ctx, &mem_2043, bytes_1954, "mem_2043"))
            return 1;
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegMap");
        
        CUdeviceptr kernel_arg_2557 = indir_mem_1933.mem;
        CUdeviceptr kernel_arg_2558 = mem_1958.mem;
        CUdeviceptr kernel_arg_2559 = res_r_mem_2032.mem;
        CUdeviceptr kernel_arg_2560 = mem_2038.mem;
        CUdeviceptr kernel_arg_2561 = mem_2043.mem;
        
        if ((((((1 && segmap_usable_groups_1599 != 0) && 1 != 0) && 1 != 0) &&
              segmap_group_sizze_1593 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = segmap_usable_groups_1599;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2554[] = {&ctx->global_failure,
                                        &ctx->failure_is_an_option,
                                        &ctx->global_failure_args,
                                        &nodes_this_lvl_1009,
                                        &pts_per_node_at_lev_1013,
                                        &kernel_arg_2557, &kernel_arg_2558,
                                        &kernel_arg_2559, &kernel_arg_2560,
                                        &kernel_arg_2561};
            int64_t time_start_2555 = 0, time_end_2556 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "segmap_1587");
                fprintf(stderr, "%d", segmap_usable_groups_1599);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segmap_group_sizze_1593);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2555 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1587, grid[0], grid[1],
                                        grid[2], segmap_group_sizze_1593, 1, 1,
                                        0, NULL, kernel_args_2554, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2556 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1587",
                        time_end_2556 - time_start_2555);
            }
        }
        ctx->failure_is_an_option = 1;
        if (memblock_unref_device(ctx, &mem_1958, "mem_1958") != 0)
            return 1;
        if (memblock_unref_device(ctx, &res_r_mem_2032, "res_r_mem_2032") != 0)
            return 1;
        
        int32_t y_1324 = sub32(nodes_this_lvl_1009, 1);
        int64_t x_1793 = add64(nodes_this_lvl_1492, y_1792);
        int64_t segmap_usable_groups_64_1795 = squot64(x_1793,
                                                       segmap_group_sizze_1791);
        int32_t segmap_usable_groups_1796 =
                sext_i64_i32(segmap_usable_groups_64_1795);
        
        if (ctx->debugging)
            fprintf(stderr, "%s\n", "\n# SegMap");
        
        CUdeviceptr kernel_arg_2565 = mem_1926.mem;
        CUdeviceptr kernel_arg_2566 = mem_1929.mem;
        CUdeviceptr kernel_arg_2567 = mem_1932.mem;
        CUdeviceptr kernel_arg_2568 = mem_1943.mem;
        CUdeviceptr kernel_arg_2569 = mem_1946.mem;
        CUdeviceptr kernel_arg_2570 = mem_2043.mem;
        
        if ((((((1 && segmap_usable_groups_1796 != 0) && 1 != 0) && 1 != 0) &&
              segmap_group_sizze_1790 != 0) && 1 != 0) && 1 != 0) {
            int perm[3] = {0, 1, 2};
            
            if (1 > 1 << 16) {
                perm[1] = perm[0];
                perm[0] = 1;
            }
            if (1 > 1 << 16) {
                perm[2] = perm[0];
                perm[0] = 2;
            }
            
            size_t grid[3];
            
            grid[perm[0]] = segmap_usable_groups_1796;
            grid[perm[1]] = 1;
            grid[perm[2]] = 1;
            
            void *kernel_args_2562[] = {&ctx->global_failure, &res_942,
                                        &nodes_this_lvl_1009,
                                        &pts_per_node_at_lev_1013, &mi_1208,
                                        &i_1213, &y_1324, &kernel_arg_2565,
                                        &kernel_arg_2566, &kernel_arg_2567,
                                        &kernel_arg_2568, &kernel_arg_2569,
                                        &kernel_arg_2570};
            int64_t time_start_2563 = 0, time_end_2564 = 0;
            
            if (ctx->debugging) {
                fprintf(stderr, "Launching %s with grid size (", "segmap_1786");
                fprintf(stderr, "%d", segmap_usable_groups_1796);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ") and block size (");
                fprintf(stderr, "%d", segmap_group_sizze_1790);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ", ");
                fprintf(stderr, "%d", 1);
                fprintf(stderr, ").\n");
                time_start_2563 = get_wall_time();
            }
            CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1786, grid[0], grid[1],
                                        grid[2], segmap_group_sizze_1790, 1, 1,
                                        0, NULL, kernel_args_2562, NULL));
            if (ctx->debugging) {
                CUDA_SUCCEED(cuCtxSynchronize());
                time_end_2564 = get_wall_time();
                fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1786",
                        time_end_2564 - time_start_2563);
            }
        }
        if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2043, "mem_2043") != 0)
            return 1;
        
        bool dim_match_1337 = res_943 == x_1014;
        bool empty_or_match_cert_1338;
        
        if (!dim_match_1337) {
            ctx->error = msgprintf("Error: %s%d%s%d%s\n\nBacktrace:\n%s",
                                   "Value of (core language) shape (", x_1014,
                                   ") cannot match shape of type `*[", res_943,
                                   "]i32`.",
                                   "-> #0  buildKDtree.fut:156:30-57\n   #1  buildKDtree.fut:181:11-50\n   #2  buildKDtree.fut:178:1-182:90\n");
            if (memblock_unref_device(ctx, &mem_2043, "mem_2043") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2038, "mem_2038") != 0)
                return 1;
            if (memblock_unref_device(ctx, &xs_expanded_mem_1965,
                                      "xs_expanded_mem_1965") != 0)
                return 1;
            if (memblock_unref_device(ctx, &xs_expanded_mem_1964,
                                      "xs_expanded_mem_1964") != 0)
                return 1;
            if (memblock_unref_device(ctx, &res_r_mem_2032, "res_r_mem_2032") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2031, "res_mem_2031") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2006, "mem_2006") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2003, "mem_2003") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2017, "mem_2017") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2013, "mem_2013") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_2009, "mem_2009") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1999, "mem_1999") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1995, "mem_1995") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1991, "mem_1991") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1987, "mem_1987") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1982, "mem_1982") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1977, "mem_1977") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1971, "mem_1971") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1963, "mem_1963") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1958, "mem_1958") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1952, "mem_1952") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1949, "mem_1949") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
                return 1;
            if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") !=
                0)
                return 1;
            if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
                return 1;
            if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
                return 1;
            if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
                return 1;
            return 1;
        }
        
        struct memblock_device indir_mem_tmp_2248;
        
        indir_mem_tmp_2248.references = NULL;
        if (memblock_set_device(ctx, &indir_mem_tmp_2248, &mem_2038,
                                "mem_2038") != 0)
            return 1;
        if (memblock_set_device(ctx, &indir_mem_1933, &indir_mem_tmp_2248,
                                "indir_mem_tmp_2248") != 0)
            return 1;
        if (memblock_unref_device(ctx, &indir_mem_tmp_2248,
                                  "indir_mem_tmp_2248") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2043, "mem_2043") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2038, "mem_2038") != 0)
            return 1;
        if (memblock_unref_device(ctx, &xs_expanded_mem_1965,
                                  "xs_expanded_mem_1965") != 0)
            return 1;
        if (memblock_unref_device(ctx, &xs_expanded_mem_1964,
                                  "xs_expanded_mem_1964") != 0)
            return 1;
        if (memblock_unref_device(ctx, &res_r_mem_2032, "res_r_mem_2032") != 0)
            return 1;
        if (memblock_unref_device(ctx, &res_mem_2031, "res_mem_2031") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2006, "mem_2006") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2003, "mem_2003") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2017, "mem_2017") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2013, "mem_2013") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_2009, "mem_2009") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1999, "mem_1999") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1995, "mem_1995") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1991, "mem_1991") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1987, "mem_1987") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1982, "mem_1982") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1977, "mem_1977") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1971, "mem_1971") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1963, "mem_1963") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1958, "mem_1958") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1952, "mem_1952") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1949, "mem_1949") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1940, "mem_1940") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1946, "mem_1946") != 0)
            return 1;
        if (memblock_unref_device(ctx, &mem_1943, "mem_1943") != 0)
            return 1;
    }
    if (memblock_set_device(ctx, &res_mem_2045, &indir_mem_1933,
                            "indir_mem_1933") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
        return 1;
    
    int32_t segmap_group_sizze_1827;
    
    segmap_group_sizze_1827 = ctx->sizes.mainzisegmap_group_sizze_1826;
    
    int64_t segmap_group_sizze_1828 = sext_i32_i64(segmap_group_sizze_1827);
    int64_t y_1829 = sub64(segmap_group_sizze_1828, 1);
    int64_t x_1830 = add64(y_1829, binop_x_1922);
    int64_t segmap_usable_groups_64_1832 = squot64(x_1830,
                                                   segmap_group_sizze_1828);
    int32_t segmap_usable_groups_1833 =
            sext_i64_i32(segmap_usable_groups_64_1832);
    struct memblock_device mem_2051;
    
    mem_2051.references = NULL;
    if (memblock_alloc_device(ctx, &mem_2051, binop_x_1922, "mem_2051"))
        return 1;
    
    struct memblock_device mem_2053;
    
    mem_2053.references = NULL;
    if (memblock_alloc_device(ctx, &mem_2053, binop_x_1922, "mem_2053"))
        return 1;
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    
    CUdeviceptr kernel_arg_2574 = res_mem_2045.mem;
    CUdeviceptr kernel_arg_2575 = mem_2051.mem;
    CUdeviceptr kernel_arg_2576 = mem_2053.mem;
    
    if ((((((1 && segmap_usable_groups_1833 != 0) && 1 != 0) && 1 != 0) &&
          segmap_group_sizze_1827 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = segmap_usable_groups_1833;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_2571[] = {&ctx->global_failure, &m_911, &res_943,
                                    &kernel_arg_2574, &kernel_arg_2575,
                                    &kernel_arg_2576};
        int64_t time_start_2572 = 0, time_end_2573 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "segmap_1823");
            fprintf(stderr, "%d", segmap_usable_groups_1833);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", segmap_group_sizze_1827);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_2572 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1823, grid[0], grid[1], grid[2],
                                    segmap_group_sizze_1827, 1, 1, 0, NULL,
                                    kernel_args_2571, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_2573 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1823",
                    time_end_2573 - time_start_2572);
        }
    }
    
    int64_t nest_sizze_1805 = mul64(d_1363, binop_x_1922);
    int32_t segmap_group_sizze_1807;
    
    segmap_group_sizze_1807 = ctx->sizes.mainzisegmap_group_sizze_1806;
    
    int64_t segmap_group_sizze_1808 = sext_i32_i64(segmap_group_sizze_1807);
    int64_t y_1809 = sub64(segmap_group_sizze_1808, 1);
    int64_t x_1810 = add64(nest_sizze_1805, y_1809);
    int64_t segmap_usable_groups_64_1812 = squot64(x_1810,
                                                   segmap_group_sizze_1808);
    int32_t segmap_usable_groups_1813 =
            sext_i64_i32(segmap_usable_groups_64_1812);
    int64_t binop_x_2058 = mul64(d_1363, binop_x_1922);
    int64_t bytes_2055 = mul64(4, binop_x_2058);
    struct memblock_device mem_2059;
    
    mem_2059.references = NULL;
    if (memblock_alloc_device(ctx, &mem_2059, bytes_2055, "mem_2059"))
        return 1;
    if (ctx->debugging)
        fprintf(stderr, "%s\n", "\n# SegMap");
    
    CUdeviceptr kernel_arg_2580 = input_mem_1905.mem;
    CUdeviceptr kernel_arg_2581 = res_mem_2045.mem;
    CUdeviceptr kernel_arg_2582 = mem_2051.mem;
    CUdeviceptr kernel_arg_2583 = mem_2053.mem;
    CUdeviceptr kernel_arg_2584 = mem_2059.mem;
    
    if ((((((1 && segmap_usable_groups_1813 != 0) && 1 != 0) && 1 != 0) &&
          segmap_group_sizze_1807 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = segmap_usable_groups_1813;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_2577[] = {&ctx->global_failure,
                                    &ctx->failure_is_an_option,
                                    &ctx->global_failure_args, &d_912, &res_943,
                                    &kernel_arg_2580, &kernel_arg_2581,
                                    &kernel_arg_2582, &kernel_arg_2583,
                                    &kernel_arg_2584};
        int64_t time_start_2578 = 0, time_end_2579 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "segmap_1801");
            fprintf(stderr, "%d", segmap_usable_groups_1813);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", segmap_group_sizze_1807);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_2578 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->segmap_1801, grid[0], grid[1], grid[2],
                                    segmap_group_sizze_1807, 1, 1, 0, NULL,
                                    kernel_args_2577, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_2579 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "segmap_1801",
                    time_end_2579 - time_start_2578);
        }
    }
    ctx->failure_is_an_option = 1;
    if (memblock_unref_device(ctx, &mem_2051, "mem_2051") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_2053, "mem_2053") != 0)
        return 1;
    out_arrsizze_2107 = res_943;
    out_arrsizze_2108 = d_912;
    out_arrsizze_2110 = res_943;
    out_arrsizze_2112 = res_942;
    out_arrsizze_2114 = res_942;
    out_arrsizze_2116 = res_942;
    if (memblock_set_device(ctx, &out_mem_2106, &mem_2059, "mem_2059") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_2109, &res_mem_2045,
                            "res_mem_2045") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_2111, &mem_1929, "mem_1929") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_2113, &mem_1926, "mem_1926") != 0)
        return 1;
    if (memblock_set_device(ctx, &out_mem_2115, &mem_1932, "mem_1932") != 0)
        return 1;
    scalar_out_2103 = res_941;
    scalar_out_2104 = res_942;
    scalar_out_2105 = res_943;
    *out_scalar_out_2414 = scalar_out_2103;
    *out_scalar_out_2415 = scalar_out_2104;
    *out_scalar_out_2416 = scalar_out_2105;
    (*out_mem_p_2417).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_2417, &out_mem_2106,
                            "out_mem_2106") != 0)
        return 1;
    *out_out_arrsizze_2418 = out_arrsizze_2107;
    *out_out_arrsizze_2419 = out_arrsizze_2108;
    (*out_mem_p_2420).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_2420, &out_mem_2109,
                            "out_mem_2109") != 0)
        return 1;
    *out_out_arrsizze_2421 = out_arrsizze_2110;
    (*out_mem_p_2422).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_2422, &out_mem_2111,
                            "out_mem_2111") != 0)
        return 1;
    *out_out_arrsizze_2423 = out_arrsizze_2112;
    (*out_mem_p_2424).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_2424, &out_mem_2113,
                            "out_mem_2113") != 0)
        return 1;
    *out_out_arrsizze_2425 = out_arrsizze_2114;
    (*out_mem_p_2426).references = NULL;
    if (memblock_set_device(ctx, &*out_mem_p_2426, &out_mem_2115,
                            "out_mem_2115") != 0)
        return 1;
    *out_out_arrsizze_2427 = out_arrsizze_2116;
    if (memblock_unref_device(ctx, &mem_2059, "mem_2059") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_2053, "mem_2053") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_2051, "mem_2051") != 0)
        return 1;
    if (memblock_unref_device(ctx, &indir_mem_1933, "indir_mem_1933") != 0)
        return 1;
    if (memblock_unref_device(ctx, &res_mem_2045, "res_mem_2045") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1932, "mem_1932") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1929, "mem_1929") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1926, "mem_1926") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1923, "mem_1923") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1920, "mem_1920") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1917, "mem_1917") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1913, "mem_1913") != 0)
        return 1;
    if (memblock_unref_device(ctx, &mem_1909, "mem_1909") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_2115, "out_mem_2115") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_2113, "out_mem_2113") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_2111, "out_mem_2111") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_2109, "out_mem_2109") != 0)
        return 1;
    if (memblock_unref_device(ctx, &out_mem_2106, "out_mem_2106") != 0)
        return 1;
    return 0;
}
static int futrts_builtinzhmap_transpose_i32(struct futhark_context *ctx,
                                             struct memblock_device destmem_0,
                                             int32_t destoffset_1,
                                             struct memblock_device srcmem_2,
                                             int32_t srcoffset_3,
                                             int32_t num_arrays_4,
                                             int32_t x_elems_5,
                                             int32_t y_elems_6,
                                             int32_t in_elems_7,
                                             int32_t out_elems_8)
{
    if (!(num_arrays_4 == 0 || (x_elems_5 == 0 || y_elems_6 == 0))) {
        int32_t muly_10 = squot32(16, x_elems_5);
        int32_t mulx_9 = squot32(16, y_elems_6);
        
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || mul32(x_elems_5,
                                                                      y_elems_6) ==
                                           in_elems_7) && (x_elems_5 == 1 ||
                                                           y_elems_6 == 1))) {
            CUDA_SUCCEED(cuMemcpy(destmem_0.mem + destoffset_1, srcmem_2.mem +
                                  srcoffset_3, mul32(in_elems_7,
                                                     (int32_t) sizeof(int32_t))));
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                unsigned int shared_sizze_2588 = 1088;
                CUdeviceptr kernel_arg_2590 = destmem_0.mem;
                CUdeviceptr kernel_arg_2591 = srcmem_2.mem;
                unsigned int shared_offset_2589 = 0;
                
                if ((((((1 && squot32(sub32(add32(x_elems_5, 16), 1), 16) !=
                         0) &&
                        squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16) !=
                        0) && num_arrays_4 != 0) && 16 != 0) && 16 != 0) && 1 !=
                    0) {
                    int perm[3] = {0, 1, 2};
                    
                    if (squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16) >
                        1 << 16) {
                        perm[1] = perm[0];
                        perm[0] = 1;
                    }
                    if (num_arrays_4 > 1 << 16) {
                        perm[2] = perm[0];
                        perm[0] = 2;
                    }
                    
                    size_t grid[3];
                    
                    grid[perm[0]] = squot32(sub32(add32(x_elems_5, 16), 1), 16);
                    grid[perm[1]] =
                        squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16);
                    grid[perm[2]] = num_arrays_4;
                    
                    void *kernel_args_2585[] = {&perm[0], &perm[1], &perm[2],
                                                &shared_offset_2589,
                                                &destoffset_1, &srcoffset_3,
                                                &num_arrays_4, &x_elems_5,
                                                &y_elems_6, &in_elems_7,
                                                &out_elems_8, &mulx_9, &muly_10,
                                                &kernel_arg_2590,
                                                &kernel_arg_2591};
                    int64_t time_start_2586 = 0, time_end_2587 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with grid size (",
                                "map_transpose_i32_low_width");
                        fprintf(stderr, "%d", squot32(sub32(add32(x_elems_5,
                                                                  16), 1), 16));
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d",
                                squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                        muly_10),
                                                                  1), muly_10),
                                                    16), 1), 16));
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", num_arrays_4);
                        fprintf(stderr, ") and block size (");
                        fprintf(stderr, "%d", 16);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", 16);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", 1);
                        fprintf(stderr, ").\n");
                        time_start_2586 = get_wall_time();
                    }
                    CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_i32_low_width,
                                                grid[0], grid[1], grid[2], 16,
                                                16, 1, 0 + (shared_sizze_2588 +
                                                            (8 -
                                                             shared_sizze_2588 %
                                                             8) % 8), NULL,
                                                kernel_args_2585, NULL));
                    if (ctx->debugging) {
                        CUDA_SUCCEED(cuCtxSynchronize());
                        time_end_2587 = get_wall_time();
                        fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                "map_transpose_i32_low_width", time_end_2587 -
                                time_start_2586);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    unsigned int shared_sizze_2595 = 1088;
                    CUdeviceptr kernel_arg_2597 = destmem_0.mem;
                    CUdeviceptr kernel_arg_2598 = srcmem_2.mem;
                    unsigned int shared_offset_2596 = 0;
                    
                    if ((((((1 &&
                             squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                     mulx_9),
                                                               1), mulx_9), 16),
                                           1), 16) != 0) &&
                            squot32(sub32(add32(y_elems_6, 16), 1), 16) != 0) &&
                           num_arrays_4 != 0) && 16 != 0) && 16 != 0) && 1 !=
                        0) {
                        int perm[3] = {0, 1, 2};
                        
                        if (squot32(sub32(add32(y_elems_6, 16), 1), 16) > 1 <<
                            16) {
                            perm[1] = perm[0];
                            perm[0] = 1;
                        }
                        if (num_arrays_4 > 1 << 16) {
                            perm[2] = perm[0];
                            perm[0] = 2;
                        }
                        
                        size_t grid[3];
                        
                        grid[perm[0]] =
                            squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                    mulx_9), 1),
                                                        mulx_9), 16), 1), 16);
                        grid[perm[1]] = squot32(sub32(add32(y_elems_6, 16), 1),
                                                16);
                        grid[perm[2]] = num_arrays_4;
                        
                        void *kernel_args_2592[] = {&perm[0], &perm[1],
                                                    &perm[2],
                                                    &shared_offset_2596,
                                                    &destoffset_1, &srcoffset_3,
                                                    &num_arrays_4, &x_elems_5,
                                                    &y_elems_6, &in_elems_7,
                                                    &out_elems_8, &mulx_9,
                                                    &muly_10, &kernel_arg_2597,
                                                    &kernel_arg_2598};
                        int64_t time_start_2593 = 0, time_end_2594 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr, "Launching %s with grid size (",
                                    "map_transpose_i32_low_height");
                            fprintf(stderr, "%d",
                                    squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                            mulx_9),
                                                                      1),
                                                                mulx_9), 16),
                                                  1), 16));
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", squot32(sub32(add32(y_elems_6,
                                                                      16), 1),
                                                          16));
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", num_arrays_4);
                            fprintf(stderr, ") and block size (");
                            fprintf(stderr, "%d", 16);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", 16);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", 1);
                            fprintf(stderr, ").\n");
                            time_start_2593 = get_wall_time();
                        }
                        CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_i32_low_height,
                                                    grid[0], grid[1], grid[2],
                                                    16, 16, 1, 0 +
                                                    (shared_sizze_2595 + (8 -
                                                                          shared_sizze_2595 %
                                                                          8) %
                                                     8), NULL, kernel_args_2592,
                                                    NULL));
                        if (ctx->debugging) {
                            CUDA_SUCCEED(cuCtxSynchronize());
                            time_end_2594 = get_wall_time();
                            fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                    "map_transpose_i32_low_height",
                                    time_end_2594 - time_start_2593);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        unsigned int shared_sizze_2602 = 1;
                        CUdeviceptr kernel_arg_2604 = destmem_0.mem;
                        CUdeviceptr kernel_arg_2605 = srcmem_2.mem;
                        unsigned int shared_offset_2603 = 0;
                        
                        if ((((((1 &&
                                 squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                 x_elems_5),
                                                           y_elems_6), 256), 1),
                                         256) != 0) && 1 != 0) && 1 != 0) &&
                              256 != 0) && 1 != 0) && 1 != 0) {
                            int perm[3] = {0, 1, 2};
                            
                            if (1 > 1 << 16) {
                                perm[1] = perm[0];
                                perm[0] = 1;
                            }
                            if (1 > 1 << 16) {
                                perm[2] = perm[0];
                                perm[0] = 2;
                            }
                            
                            size_t grid[3];
                            
                            grid[perm[0]] =
                                squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                x_elems_5),
                                                          y_elems_6), 256), 1),
                                        256);
                            grid[perm[1]] = 1;
                            grid[perm[2]] = 1;
                            
                            void *kernel_args_2599[] = {&shared_offset_2603,
                                                        &destoffset_1,
                                                        &srcoffset_3,
                                                        &num_arrays_4,
                                                        &x_elems_5, &y_elems_6,
                                                        &in_elems_7,
                                                        &out_elems_8, &mulx_9,
                                                        &muly_10,
                                                        &kernel_arg_2604,
                                                        &kernel_arg_2605};
                            int64_t time_start_2600 = 0, time_end_2601 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr, "Launching %s with grid size (",
                                        "map_transpose_i32_small");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                        x_elems_5),
                                                                  y_elems_6),
                                                            256), 1), 256));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ") and block size (");
                                fprintf(stderr, "%d", 256);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ").\n");
                                time_start_2600 = get_wall_time();
                            }
                            CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_i32_small,
                                                        grid[0], grid[1],
                                                        grid[2], 256, 1, 1, 0 +
                                                        (shared_sizze_2602 +
                                                         (8 -
                                                          shared_sizze_2602 %
                                                          8) % 8), NULL,
                                                        kernel_args_2599,
                                                        NULL));
                            if (ctx->debugging) {
                                CUDA_SUCCEED(cuCtxSynchronize());
                                time_end_2601 = get_wall_time();
                                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                        "map_transpose_i32_small",
                                        time_end_2601 - time_start_2600);
                            }
                        }
                    } else {
                        unsigned int shared_sizze_2609 = 4224;
                        CUdeviceptr kernel_arg_2611 = destmem_0.mem;
                        CUdeviceptr kernel_arg_2612 = srcmem_2.mem;
                        unsigned int shared_offset_2610 = 0;
                        
                        if ((((((1 && squot32(sub32(add32(x_elems_5, 32), 1),
                                              32) != 0) &&
                                squot32(sub32(add32(y_elems_6, 32), 1), 32) !=
                                0) && num_arrays_4 != 0) && 32 != 0) && 8 !=
                             0) && 1 != 0) {
                            int perm[3] = {0, 1, 2};
                            
                            if (squot32(sub32(add32(y_elems_6, 32), 1), 32) >
                                1 << 16) {
                                perm[1] = perm[0];
                                perm[0] = 1;
                            }
                            if (num_arrays_4 > 1 << 16) {
                                perm[2] = perm[0];
                                perm[0] = 2;
                            }
                            
                            size_t grid[3];
                            
                            grid[perm[0]] = squot32(sub32(add32(x_elems_5, 32),
                                                          1), 32);
                            grid[perm[1]] = squot32(sub32(add32(y_elems_6, 32),
                                                          1), 32);
                            grid[perm[2]] = num_arrays_4;
                            
                            void *kernel_args_2606[] = {&perm[0], &perm[1],
                                                        &perm[2],
                                                        &shared_offset_2610,
                                                        &destoffset_1,
                                                        &srcoffset_3,
                                                        &num_arrays_4,
                                                        &x_elems_5, &y_elems_6,
                                                        &in_elems_7,
                                                        &out_elems_8, &mulx_9,
                                                        &muly_10,
                                                        &kernel_arg_2611,
                                                        &kernel_arg_2612};
                            int64_t time_start_2607 = 0, time_end_2608 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr, "Launching %s with grid size (",
                                        "map_transpose_i32");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(x_elems_5, 32), 1),
                                                32));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(y_elems_6, 32), 1),
                                                32));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", num_arrays_4);
                                fprintf(stderr, ") and block size (");
                                fprintf(stderr, "%d", 32);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 8);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ").\n");
                                time_start_2607 = get_wall_time();
                            }
                            CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_i32,
                                                        grid[0], grid[1],
                                                        grid[2], 32, 8, 1, 0 +
                                                        (shared_sizze_2609 +
                                                         (8 -
                                                          shared_sizze_2609 %
                                                          8) % 8), NULL,
                                                        kernel_args_2606,
                                                        NULL));
                            if (ctx->debugging) {
                                CUDA_SUCCEED(cuCtxSynchronize());
                                time_end_2608 = get_wall_time();
                                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                        "map_transpose_i32", time_end_2608 -
                                        time_start_2607);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts_builtinzhreplicate_i32(struct futhark_context *ctx,
                                         struct memblock_device mem_2239,
                                         int32_t num_elems_2240,
                                         int32_t val_2241)
{
    int32_t group_sizze_2246;
    
    group_sizze_2246 = ctx->sizes.mainzigroup_sizze_2246;
    
    int32_t num_groups_2247;
    
    num_groups_2247 = squot32(sub32(add32(num_elems_2240,
                                          sext_i32_i32(group_sizze_2246)), 1),
                              sext_i32_i32(group_sizze_2246));
    
    CUdeviceptr kernel_arg_2616 = mem_2239.mem;
    
    if ((((((1 && num_groups_2247 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_2246 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_2247;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_2613[] = {&kernel_arg_2616, &num_elems_2240,
                                    &val_2241};
        int64_t time_start_2614 = 0, time_end_2615 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_2243");
            fprintf(stderr, "%d", num_groups_2247);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_2246);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_2614 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_2243, grid[0], grid[1],
                                    grid[2], group_sizze_2246, 1, 1, 0, NULL,
                                    kernel_args_2613, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_2615 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_2243",
                    time_end_2615 - time_start_2614);
        }
    }
    return 0;
}
static int futrts_builtinzhreplicate_f32(struct futhark_context *ctx,
                                         struct memblock_device mem_2230,
                                         int32_t num_elems_2231, float val_2232)
{
    int32_t group_sizze_2237;
    
    group_sizze_2237 = ctx->sizes.mainzigroup_sizze_2237;
    
    int32_t num_groups_2238;
    
    num_groups_2238 = squot32(sub32(add32(num_elems_2231,
                                          sext_i32_i32(group_sizze_2237)), 1),
                              sext_i32_i32(group_sizze_2237));
    
    CUdeviceptr kernel_arg_2620 = mem_2230.mem;
    
    if ((((((1 && num_groups_2238 != 0) && 1 != 0) && 1 != 0) &&
          group_sizze_2237 != 0) && 1 != 0) && 1 != 0) {
        int perm[3] = {0, 1, 2};
        
        if (1 > 1 << 16) {
            perm[1] = perm[0];
            perm[0] = 1;
        }
        if (1 > 1 << 16) {
            perm[2] = perm[0];
            perm[0] = 2;
        }
        
        size_t grid[3];
        
        grid[perm[0]] = num_groups_2238;
        grid[perm[1]] = 1;
        grid[perm[2]] = 1;
        
        void *kernel_args_2617[] = {&kernel_arg_2620, &num_elems_2231,
                                    &val_2232};
        int64_t time_start_2618 = 0, time_end_2619 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with grid size (", "replicate_2234");
            fprintf(stderr, "%d", num_groups_2238);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ") and block size (");
            fprintf(stderr, "%d", group_sizze_2237);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ", ");
            fprintf(stderr, "%d", 1);
            fprintf(stderr, ").\n");
            time_start_2618 = get_wall_time();
        }
        CUDA_SUCCEED(cuLaunchKernel(ctx->replicate_2234, grid[0], grid[1],
                                    grid[2], group_sizze_2237, 1, 1, 0, NULL,
                                    kernel_args_2617, NULL));
        if (ctx->debugging) {
            CUDA_SUCCEED(cuCtxSynchronize());
            time_end_2619 = get_wall_time();
            fprintf(stderr, "Kernel %s runtime: %ldus\n", "replicate_2234",
                    time_end_2619 - time_start_2618);
        }
    }
    return 0;
}
static int futrts_builtinzhmap_transpose_f32(struct futhark_context *ctx,
                                             struct memblock_device destmem_0,
                                             int32_t destoffset_1,
                                             struct memblock_device srcmem_2,
                                             int32_t srcoffset_3,
                                             int32_t num_arrays_4,
                                             int32_t x_elems_5,
                                             int32_t y_elems_6,
                                             int32_t in_elems_7,
                                             int32_t out_elems_8)
{
    if (!(num_arrays_4 == 0 || (x_elems_5 == 0 || y_elems_6 == 0))) {
        int32_t muly_10 = squot32(16, x_elems_5);
        int32_t mulx_9 = squot32(16, y_elems_6);
        
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || mul32(x_elems_5,
                                                                      y_elems_6) ==
                                           in_elems_7) && (x_elems_5 == 1 ||
                                                           y_elems_6 == 1))) {
            CUDA_SUCCEED(cuMemcpy(destmem_0.mem + destoffset_1, srcmem_2.mem +
                                  srcoffset_3, mul32(in_elems_7,
                                                     (int32_t) sizeof(float))));
        } else {
            if (sle32(x_elems_5, 8) && slt32(16, y_elems_6)) {
                unsigned int shared_sizze_2624 = 1088;
                CUdeviceptr kernel_arg_2626 = destmem_0.mem;
                CUdeviceptr kernel_arg_2627 = srcmem_2.mem;
                unsigned int shared_offset_2625 = 0;
                
                if ((((((1 && squot32(sub32(add32(x_elems_5, 16), 1), 16) !=
                         0) &&
                        squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16) !=
                        0) && num_arrays_4 != 0) && 16 != 0) && 16 != 0) && 1 !=
                    0) {
                    int perm[3] = {0, 1, 2};
                    
                    if (squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16) >
                        1 << 16) {
                        perm[1] = perm[0];
                        perm[0] = 1;
                    }
                    if (num_arrays_4 > 1 << 16) {
                        perm[2] = perm[0];
                        perm[0] = 2;
                    }
                    
                    size_t grid[3];
                    
                    grid[perm[0]] = squot32(sub32(add32(x_elems_5, 16), 1), 16);
                    grid[perm[1]] =
                        squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                muly_10), 1),
                                                    muly_10), 16), 1), 16);
                    grid[perm[2]] = num_arrays_4;
                    
                    void *kernel_args_2621[] = {&perm[0], &perm[1], &perm[2],
                                                &shared_offset_2625,
                                                &destoffset_1, &srcoffset_3,
                                                &num_arrays_4, &x_elems_5,
                                                &y_elems_6, &in_elems_7,
                                                &out_elems_8, &mulx_9, &muly_10,
                                                &kernel_arg_2626,
                                                &kernel_arg_2627};
                    int64_t time_start_2622 = 0, time_end_2623 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with grid size (",
                                "map_transpose_f32_low_width");
                        fprintf(stderr, "%d", squot32(sub32(add32(x_elems_5,
                                                                  16), 1), 16));
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d",
                                squot32(sub32(add32(squot32(sub32(add32(y_elems_6,
                                                                        muly_10),
                                                                  1), muly_10),
                                                    16), 1), 16));
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", num_arrays_4);
                        fprintf(stderr, ") and block size (");
                        fprintf(stderr, "%d", 16);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", 16);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%d", 1);
                        fprintf(stderr, ").\n");
                        time_start_2622 = get_wall_time();
                    }
                    CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_f32_low_width,
                                                grid[0], grid[1], grid[2], 16,
                                                16, 1, 0 + (shared_sizze_2624 +
                                                            (8 -
                                                             shared_sizze_2624 %
                                                             8) % 8), NULL,
                                                kernel_args_2621, NULL));
                    if (ctx->debugging) {
                        CUDA_SUCCEED(cuCtxSynchronize());
                        time_end_2623 = get_wall_time();
                        fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                "map_transpose_f32_low_width", time_end_2623 -
                                time_start_2622);
                    }
                }
            } else {
                if (sle32(y_elems_6, 8) && slt32(16, x_elems_5)) {
                    unsigned int shared_sizze_2631 = 1088;
                    CUdeviceptr kernel_arg_2633 = destmem_0.mem;
                    CUdeviceptr kernel_arg_2634 = srcmem_2.mem;
                    unsigned int shared_offset_2632 = 0;
                    
                    if ((((((1 &&
                             squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                     mulx_9),
                                                               1), mulx_9), 16),
                                           1), 16) != 0) &&
                            squot32(sub32(add32(y_elems_6, 16), 1), 16) != 0) &&
                           num_arrays_4 != 0) && 16 != 0) && 16 != 0) && 1 !=
                        0) {
                        int perm[3] = {0, 1, 2};
                        
                        if (squot32(sub32(add32(y_elems_6, 16), 1), 16) > 1 <<
                            16) {
                            perm[1] = perm[0];
                            perm[0] = 1;
                        }
                        if (num_arrays_4 > 1 << 16) {
                            perm[2] = perm[0];
                            perm[0] = 2;
                        }
                        
                        size_t grid[3];
                        
                        grid[perm[0]] =
                            squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                    mulx_9), 1),
                                                        mulx_9), 16), 1), 16);
                        grid[perm[1]] = squot32(sub32(add32(y_elems_6, 16), 1),
                                                16);
                        grid[perm[2]] = num_arrays_4;
                        
                        void *kernel_args_2628[] = {&perm[0], &perm[1],
                                                    &perm[2],
                                                    &shared_offset_2632,
                                                    &destoffset_1, &srcoffset_3,
                                                    &num_arrays_4, &x_elems_5,
                                                    &y_elems_6, &in_elems_7,
                                                    &out_elems_8, &mulx_9,
                                                    &muly_10, &kernel_arg_2633,
                                                    &kernel_arg_2634};
                        int64_t time_start_2629 = 0, time_end_2630 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr, "Launching %s with grid size (",
                                    "map_transpose_f32_low_height");
                            fprintf(stderr, "%d",
                                    squot32(sub32(add32(squot32(sub32(add32(x_elems_5,
                                                                            mulx_9),
                                                                      1),
                                                                mulx_9), 16),
                                                  1), 16));
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", squot32(sub32(add32(y_elems_6,
                                                                      16), 1),
                                                          16));
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", num_arrays_4);
                            fprintf(stderr, ") and block size (");
                            fprintf(stderr, "%d", 16);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", 16);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%d", 1);
                            fprintf(stderr, ").\n");
                            time_start_2629 = get_wall_time();
                        }
                        CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_f32_low_height,
                                                    grid[0], grid[1], grid[2],
                                                    16, 16, 1, 0 +
                                                    (shared_sizze_2631 + (8 -
                                                                          shared_sizze_2631 %
                                                                          8) %
                                                     8), NULL, kernel_args_2628,
                                                    NULL));
                        if (ctx->debugging) {
                            CUDA_SUCCEED(cuCtxSynchronize());
                            time_end_2630 = get_wall_time();
                            fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                    "map_transpose_f32_low_height",
                                    time_end_2630 - time_start_2629);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, 8) && sle32(y_elems_6, 8)) {
                        unsigned int shared_sizze_2638 = 1;
                        CUdeviceptr kernel_arg_2640 = destmem_0.mem;
                        CUdeviceptr kernel_arg_2641 = srcmem_2.mem;
                        unsigned int shared_offset_2639 = 0;
                        
                        if ((((((1 &&
                                 squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                 x_elems_5),
                                                           y_elems_6), 256), 1),
                                         256) != 0) && 1 != 0) && 1 != 0) &&
                              256 != 0) && 1 != 0) && 1 != 0) {
                            int perm[3] = {0, 1, 2};
                            
                            if (1 > 1 << 16) {
                                perm[1] = perm[0];
                                perm[0] = 1;
                            }
                            if (1 > 1 << 16) {
                                perm[2] = perm[0];
                                perm[0] = 2;
                            }
                            
                            size_t grid[3];
                            
                            grid[perm[0]] =
                                squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                x_elems_5),
                                                          y_elems_6), 256), 1),
                                        256);
                            grid[perm[1]] = 1;
                            grid[perm[2]] = 1;
                            
                            void *kernel_args_2635[] = {&shared_offset_2639,
                                                        &destoffset_1,
                                                        &srcoffset_3,
                                                        &num_arrays_4,
                                                        &x_elems_5, &y_elems_6,
                                                        &in_elems_7,
                                                        &out_elems_8, &mulx_9,
                                                        &muly_10,
                                                        &kernel_arg_2640,
                                                        &kernel_arg_2641};
                            int64_t time_start_2636 = 0, time_end_2637 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr, "Launching %s with grid size (",
                                        "map_transpose_f32_small");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(mul32(mul32(num_arrays_4,
                                                                        x_elems_5),
                                                                  y_elems_6),
                                                            256), 1), 256));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ") and block size (");
                                fprintf(stderr, "%d", 256);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ").\n");
                                time_start_2636 = get_wall_time();
                            }
                            CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_f32_small,
                                                        grid[0], grid[1],
                                                        grid[2], 256, 1, 1, 0 +
                                                        (shared_sizze_2638 +
                                                         (8 -
                                                          shared_sizze_2638 %
                                                          8) % 8), NULL,
                                                        kernel_args_2635,
                                                        NULL));
                            if (ctx->debugging) {
                                CUDA_SUCCEED(cuCtxSynchronize());
                                time_end_2637 = get_wall_time();
                                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                        "map_transpose_f32_small",
                                        time_end_2637 - time_start_2636);
                            }
                        }
                    } else {
                        unsigned int shared_sizze_2645 = 4224;
                        CUdeviceptr kernel_arg_2647 = destmem_0.mem;
                        CUdeviceptr kernel_arg_2648 = srcmem_2.mem;
                        unsigned int shared_offset_2646 = 0;
                        
                        if ((((((1 && squot32(sub32(add32(x_elems_5, 32), 1),
                                              32) != 0) &&
                                squot32(sub32(add32(y_elems_6, 32), 1), 32) !=
                                0) && num_arrays_4 != 0) && 32 != 0) && 8 !=
                             0) && 1 != 0) {
                            int perm[3] = {0, 1, 2};
                            
                            if (squot32(sub32(add32(y_elems_6, 32), 1), 32) >
                                1 << 16) {
                                perm[1] = perm[0];
                                perm[0] = 1;
                            }
                            if (num_arrays_4 > 1 << 16) {
                                perm[2] = perm[0];
                                perm[0] = 2;
                            }
                            
                            size_t grid[3];
                            
                            grid[perm[0]] = squot32(sub32(add32(x_elems_5, 32),
                                                          1), 32);
                            grid[perm[1]] = squot32(sub32(add32(y_elems_6, 32),
                                                          1), 32);
                            grid[perm[2]] = num_arrays_4;
                            
                            void *kernel_args_2642[] = {&perm[0], &perm[1],
                                                        &perm[2],
                                                        &shared_offset_2646,
                                                        &destoffset_1,
                                                        &srcoffset_3,
                                                        &num_arrays_4,
                                                        &x_elems_5, &y_elems_6,
                                                        &in_elems_7,
                                                        &out_elems_8, &mulx_9,
                                                        &muly_10,
                                                        &kernel_arg_2647,
                                                        &kernel_arg_2648};
                            int64_t time_start_2643 = 0, time_end_2644 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr, "Launching %s with grid size (",
                                        "map_transpose_f32");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(x_elems_5, 32), 1),
                                                32));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d",
                                        squot32(sub32(add32(y_elems_6, 32), 1),
                                                32));
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", num_arrays_4);
                                fprintf(stderr, ") and block size (");
                                fprintf(stderr, "%d", 32);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 8);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%d", 1);
                                fprintf(stderr, ").\n");
                                time_start_2643 = get_wall_time();
                            }
                            CUDA_SUCCEED(cuLaunchKernel(ctx->map_transpose_f32,
                                                        grid[0], grid[1],
                                                        grid[2], 32, 8, 1, 0 +
                                                        (shared_sizze_2645 +
                                                         (8 -
                                                          shared_sizze_2645 %
                                                          8) % 8), NULL,
                                                        kernel_args_2642,
                                                        NULL));
                            if (ctx->debugging) {
                                CUDA_SUCCEED(cuCtxSynchronize());
                                time_end_2644 = get_wall_time();
                                fprintf(stderr, "Kernel %s runtime: %ldus\n",
                                        "map_transpose_f32", time_end_2644 -
                                        time_start_2643);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
struct futhark_f32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, (size_t) dim0 *
                              sizeof(float)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(float),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, (size_t) dim0 *
                          sizeof(float)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0,
                              (size_t) arr->shape[0] * sizeof(float)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                      struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_i32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, (size_t) dim0 *
                              sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) dim0 * sizeof(int32_t),
                              "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, (size_t) dim0 *
                          sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0,
                              (size_t) arr->shape[0] * sizeof(int32_t)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                      struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    (void) ctx;
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    CUDA_SUCCEED(cuMemcpyHtoD(arr->mem.mem + 0, data + 0, (size_t) (dim0 *
                                                                    dim1) *
                              sizeof(float)));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              CUdeviceptr data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc_device(ctx, &arr->mem, (size_t) (dim0 * dim1) *
                              sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    CUDA_SUCCEED(cuMemcpy(arr->mem.mem + 0, data + offset, (size_t) (dim0 *
                                                                     dim1) *
                          sizeof(float)));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref_device(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    CUDA_SUCCEED(cuMemcpyDtoH(data + 0, arr->mem.mem + 0,
                              (size_t) (arr->shape[0] * arr->shape[1]) *
                              sizeof(float)));
    lock_unlock(&ctx->lock);
    return 0;
}
CUdeviceptr futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                      struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    (void) ctx;
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx, int32_t *out0,
                       int32_t *out1, int32_t *out2,
                       struct futhark_f32_2d **out3,
                       struct futhark_i32_1d **out4,
                       struct futhark_i32_1d **out5,
                       struct futhark_f32_1d **out6,
                       struct futhark_i32_1d **out7, const int32_t in0, const
                       struct futhark_f32_2d *in1)
{
    struct memblock_device input_mem_1905;
    
    input_mem_1905.references = NULL;
    
    int32_t m_911;
    int32_t d_912;
    int32_t defppl_913;
    int32_t scalar_out_2103;
    int32_t scalar_out_2104;
    int32_t scalar_out_2105;
    struct memblock_device out_mem_2106;
    
    out_mem_2106.references = NULL;
    
    int32_t out_arrsizze_2107;
    int32_t out_arrsizze_2108;
    struct memblock_device out_mem_2109;
    
    out_mem_2109.references = NULL;
    
    int32_t out_arrsizze_2110;
    struct memblock_device out_mem_2111;
    
    out_mem_2111.references = NULL;
    
    int32_t out_arrsizze_2112;
    struct memblock_device out_mem_2113;
    
    out_mem_2113.references = NULL;
    
    int32_t out_arrsizze_2114;
    struct memblock_device out_mem_2115;
    
    out_mem_2115.references = NULL;
    
    int32_t out_arrsizze_2116;
    
    lock_lock(&ctx->lock);
    defppl_913 = in0;
    input_mem_1905 = in1->mem;
    m_911 = in1->shape[0];
    d_912 = in1->shape[1];
    
    int ret = futrts_main(ctx, &scalar_out_2103, &scalar_out_2104,
                          &scalar_out_2105, &out_mem_2106, &out_arrsizze_2107,
                          &out_arrsizze_2108, &out_mem_2109, &out_arrsizze_2110,
                          &out_mem_2111, &out_arrsizze_2112, &out_mem_2113,
                          &out_arrsizze_2114, &out_mem_2115, &out_arrsizze_2116,
                          input_mem_1905, m_911, d_912, defppl_913);
    
    if (ret == 0) {
        *out0 = scalar_out_2103;
        *out1 = scalar_out_2104;
        *out2 = scalar_out_2105;
        assert((*out3 =
                (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d))) !=
            NULL);
        (*out3)->mem = out_mem_2106;
        (*out3)->shape[0] = out_arrsizze_2107;
        (*out3)->shape[1] = out_arrsizze_2108;
        assert((*out4 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out4)->mem = out_mem_2109;
        (*out4)->shape[0] = out_arrsizze_2110;
        assert((*out5 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out5)->mem = out_mem_2111;
        (*out5)->shape[0] = out_arrsizze_2112;
        assert((*out6 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out6)->mem = out_mem_2113;
        (*out6)->shape[0] = out_arrsizze_2114;
        assert((*out7 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out7)->mem = out_mem_2115;
        (*out7)->shape[0] = out_arrsizze_2116;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
