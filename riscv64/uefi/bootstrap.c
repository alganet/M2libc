/* Copyright (C) 2022 Andrius Štikonas
 * Copyright (C) 2026 Alexandre Gomes Gaigalas
 * This file is part of M2-Planet.
 *
 * M2-Planet is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * M2-Planet is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with M2-Planet.  If not, see <http://www.gnu.org/licenses/>.
 */

/* ====================================================================
 * riscv64 UEFI bootstrap.c -- minimal libc + UEFI service wrappers
 * ====================================================================
 *
 * What this file provides:
 *   The minimum C surface that cc_riscv64 (a self-hosted compiler
 *   built from a hand-written .M1) needs in order to compile larger
 *   programs (M2-Planet, mescc-tools-extra, etc.) inside the UEFI
 *   environment. We can't use a preprocessor here -- cc_riscv64 has
 *   no #include support -- so every type, constant, and prototype
 *   that the compiled program may reference lives in a single flat
 *   file, in declaration order.
 *
 * Layout (in the order things appear below):
 *
 *   1. Constants  -- stdin/stdout/stderr fds, EOF, EXIT_*, TRUE/FALSE,
 *      PAGE_SIZE/PAGE_NUM/USER_STACK_SIZE, UEFI flags.
 *   2. Globals    -- UEFI image handle, malloc state, argc/argv.
 *   3. UEFI types -- a thin transcription of the EFI_*_PROTOCOL and
 *      EFI_BOOT_SERVICES table layouts.
 *   4. Asm wrappers -- one C function per UEFI service we call. Each
 *      reads its arguments from the M2-Planet save frame at fp-N
 *      and shuffles them into the riscv64 ABI registers a0-a7,
 *      then `jalr ra, t0` to the service pointer. The asm idiom
 *      is documented below; once one wrapper is understood, the
 *      rest are trivial.
 *   5. Higher-level libc -- fgetc/fread/fputc/fwrite/fputs/fopen/
 *      fclose/strlen, malloc/calloc/free, memset, exit.
 *   6. UEFI <-> POSIX glue -- UCS-2 <-> ASCII, '/' <-> '\\' path
 *      conversion, command-line parsing.
 *   7. _init / _cleanup -- called by libc-{core,full}.M1's _start.
 *      _init opens the FS root (via SIMPLE_FILE_SYSTEM_PROTOCOL),
 *      parses LoadOptions, allocates the user stack. _cleanup
 *      closes everything and frees pages.
 *
 * cc_riscv64 quirks compensated for here:
 *
 *   * cc_riscv64 emits `ld` (8-byte load) for ALL struct field
 *     accesses, even UINT32 fields. UEFI's load_options_size is
 *     UINT32 (4 bytes + 4 bytes of struct padding), so a naive
 *     `image->load_options_size` reads the padding bytes as the
 *     upper half of a 64-bit value. _clamp_u32() (slli #32 +
 *     srli #32) masks to the low 32 bits. The same pattern shows
 *     up wherever we read a UEFI-declared UINT32.
 *
 *   * cc_riscv64 doesn't accept addresses in 64-bit-constant form,
 *     so the GUIDs are assembled from two 32-bit halves with an
 *     offset trick to avoid the implicit-signed-32-bit issue
 *     (see EFI_LOADED_IMAGE_PROTOCOL_GUID setup in _init).
 */

/* riscv64 UEFI bootstrap for cc_riscv64 (no preprocessor) */

/* === Section 1: Constants ===========================================*/

/* POSIX file-descriptor numbers. Used as sentinels by fputc to route
 * stdout/stderr writes through ConOut (UEFI's text console) rather
 * than treating them as file pointers. */
enum
{
	stdin = 0,
	stdout = 1,
	stderr = 2,
};

/* Standard libc constants. EOF is 0xFFFFFFFF (== -1 in 32-bit twos-
 * complement) so it sign-extends cleanly when promoted to long. */
enum
{
	EOF = 0xFFFFFFFF,
	NULL = 0,
};

enum
{
	EXIT_FAILURE = 1,
	EXIT_SUCCESS = 0,
};

enum
{
	TRUE = 1,
	FALSE = 0,
};

/* UEFI heap & stack tunables.
 *   PAGE_SIZE     -- UEFI's AllocatePages granularity (4 KiB).
 *   PAGE_NUM      -- number of pages malloc claims at first use:
 *                     65536 x 4 KiB = 256 MiB.
 *                   This was 16384 (64 MiB) until the M2-Planet
 *                   self-compile started OOM'ing -- the resulting
 *                   AST + symbol table for mes.c + bootstrap.c +
 *                   crt1.c + 50 other inputs needs more than 64 MiB.
 *   USER_STACK_SIZE -- 8 MiB user stack. UEFI's own stack is
 *                     typically 64 KiB which the compiler exhausts
 *                     during deep recursion, so libc-core.M1's
 *                     exit_uefi_stack swaps to this larger one. */
enum
{
	PAGE_SIZE = 4096,
	PAGE_NUM = 65536, /* 256 MiB heap (was 64 MiB; M2-Planet self-compile needs more) */
	USER_STACK_SIZE = 8388608,
};

/* UEFI service constants used in OpenProtocol/Open/AllocatePages. */
enum
{
	EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL = 1,
	EFI_FILE_MODE_READ = 1,
	EFI_FILE_MODE_WRITE = 2,
	EFI_FILE_READ_ONLY = 1,
	EFI_ALLOCATE_ANY_PAGES = 0,
	EFI_LOADER_DATA = 2,
};

void exit(unsigned value);

/* === Section 2: Globals =============================================
 * Set by libc-core.M1's _start (image_handle, system) and _init
 * (root_device, user_stack, malloc state, argc/argv). C code reaches
 * them by name; M1 reaches them via :GLOBAL__<name> labels. */
void* _image_handle;
void* _root_device;
void* __user_stack;
void* _malloc_start;     /* base of malloc heap (for free_pages) */
long _malloc_ptr;        /* next-free-byte cursor inside the heap */
long _brk_ptr;           /* one-past-end of allocated heap */
int _argc;
char** _argv;

/* === Section 3: UEFI types ==========================================
 * Faithful transcription of the EFI_*_PROTOCOL and EFI_BOOT_SERVICES
 * struct layouts from the UEFI 2.x spec. Field order is FIXED (UEFI
 * passes pointers to these structs and expects offsets to match).
 * cc_riscv64 only supports `void*` and basic int types here, so all
 * function-pointer slots are typed as `void*` and called via the
 * inline-asm wrappers in Section 4. */

struct efi_simple_text_output_protocol
{
	void* reset;
	void* output_string;
	void* test_string;
	void* query_mode;
	void* set_mode;
	void* set_attribute;
	void* clear_screen;
	void* set_cursor;
	void* enable_cursor;
	void* mode;
};

struct efi_table_header
{
	unsigned signature;
	unsigned revision_and_header_size;
	unsigned crc32_and_reserved;
};

struct efi_boot_table
{
	struct efi_table_header header;

	/* Task Priority Services */
	void* raise_tpl;
	void* restore_tpl;

	/* Memory Services */
	void* allocate_pages;
	void* free_pages;
	void* get_memory_map;
	void* allocate_pool;
	void* free_pool;

	/* Event & Timer Services */
	void* create_event;
	void* set_timer;
	void* wait_for_event;
	void* signal_event;
	void* close_event;
	void* check_event;

	/* Protocol Handler Services */
	void* install_protocol_interface;
	void* reinstall_protocol_interface;
	void* uninstall_protocol_interface;
	void* handle_protocol;
	void* reserved;
	void* register_protocol_notify;
	void* locate_handle;
	void* locate_device_path;
	void* install_configuration_table;

	/* Image Services */
	void* load_image;
	void* start_image;
	void* exit;
	void* unload_image;
	void* exit_boot_services;

	/* Miscellaneous Services */
	void* get_next_monotonic_count;
	void* stall;
	void* set_watchdog_timer;

	/* DriverSupport Services */
	void* connect_controller;
	void* disconnect_controller;

	/* Open and Close Protocol Services */
	void* open_protocol;
	void* close_protocol;
	void* open_protocol_information;

	/* Library Services */
	void* protocols_per_handle;
	void* locate_handle_buffer;
	void* locate_protocol;
	void* install_multiple_protocol_interfaces;
	void* uninstall_multiple_protocol_interfaces;

	/* 32-bit CRC Services */
	void* copy_mem;
	void* set_mem;
	void* create_event_ex;
};

struct efi_system_table
{
	struct efi_table_header header;

	char* firmware_vendor;
	unsigned firmware_revision;
	void* console_in_handle;
	void* con_in;
	void* console_out_handle;
	struct efi_simple_text_output_protocol* con_out;
	void *standard_error_handle;
	struct efi_simple_text_output_protocol* std_err;
	void *runtime_services;
	struct efi_boot_table* boot_services;
	unsigned number_table_entries;
	void *configuration_table;
};
struct efi_system_table* _system;

struct efi_guid
{
	unsigned data1;
	unsigned data2;
};
struct efi_guid* EFI_LOADED_IMAGE_PROTOCOL_GUID;
struct efi_guid* EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID;

struct efi_loaded_image_protocol
{
	unsigned revision;
	void* parent;
	void* system;

	void* device;
	void* filepath;
	void* reserved;

	/* Image's load options */
	unsigned load_options_size;
	void* load_options;

	/* Location of the image in memory */
	void* image_base;
	unsigned image_size;
	unsigned image_code_type;
	unsigned image_data_type;
	void* unload;
};

struct efi_simple_file_system_protocol
{
	unsigned revision;
	void* open_volume;
};

struct efi_file_protocol
{
	unsigned revision;
	void* open;
	void* close;
	void* delete;
	void* read;
	void* write;
	void* get_position;
	void* set_position;
	void* get_info;
	void* set_info;
	void* flush;
	void* open_ex;
	void* read_ex;
	void* write_ex;
	void* flush_ex;
};
struct efi_file_protocol* _rootdir;

/* === Section 4: UEFI service wrappers (inline asm) ==================
 *
 * Each wrapper function has a body of ONLY M1 inline asm -- no C
 * statements. cc_riscv64 turns `asm("...")` into the literal M1
 * tokens passed verbatim, so we can hand-roll the riscv64 calling
 * convention and the UEFI service-pointer dispatch.
 *
 * The standard idiom in every wrapper:
 *
 *   1. Load each declared C parameter from its M2-Planet save-frame
 *      slot at fp-N (8, 16, 24, ...) into the corresponding riscv64
 *      argument register a0..a7. The order matches the parameter
 *      list as declared at the C function signature above.
 *
 *   2. The LAST parameter is always `FUNCTION svc` -- the actual
 *      UEFI service-pointer, threaded through from C so each call
 *      site reaches the correct slot in EFI_BOOT_SERVICES /
 *      EFI_FILE_PROTOCOL. We load it into t0.
 *
 *   3. Some wrappers reserve scratch space for output-pointer args:
 *      UEFI's protocol-handle and buffer arguments are pass-by-
 *      pointer (UEFI returns through the pointer), so we sub-sp to
 *      get a temporary slot, take its address into the right aN
 *      register, then load the result back into a0 after the call.
 *
 *   4. ra is stashed (sp -= 8 ; sd ra) before `jalr ra, t0` and
 *      reloaded after, because jalr clobbers ra.
 *
 *   5. For wrappers with a meaningful return, the final block reads
 *      a0 from wherever it landed (often an output pointer slot)
 *      and adjusts sp back to its caller's value.
 *
 * RISC-V UEFI calling convention: args in a0-a7, function pointer
 * in t0, jalr ra, t0.
 */

/* _read -- EFI_FILE_PROTOCOL.Read(This, BufferSize*, Buffer*).
 * Reads up to `size` bytes (we only ever pass size=1 here) into a
 * stack-temp byte and returns it as the function's char result.
 * Returns -1 (== EOF after promotion) when read returns 0 bytes. */
char _read(FILE* f, unsigned size, FUNCTION read)
{
	/* EFI File.Read(This=a0, BufferSize*=a1, Buffer*=a2) */
	asm("rd_a0 rs1_fp !-8 ld"           /* a0 = file */
	    "rd_sp rs1_sp !-16 addi"        /* sp -= 16: [buffer@sp+0, size@sp+8] */
	    "rs1_sp rs2_zero sd"            /* *(sp+0) = 0 (byte buffer) */
	    "rd_t0 rs1_fp !-16 ld"          /* t0 = size value */
	    "rs1_sp rs2_t0 @8 sd"           /* *(sp+8) = size */
	    "rd_a1 rs1_sp !8 addi"          /* a1 = sp+8 = &size */
	    "rd_a2 rs1_sp mv"               /* a2 = sp = &buffer */
	    "rd_t0 rs1_fp !-24 ld"          /* t0 = file->read fn */
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"             /* call read(file, &size, &buffer) */
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi"
	    "rd_t0 rs1_sp !8 ld"            /* t0 = bytes actually read */
	    "rd_a0 rs1_sp lbu"              /* a0 = byte (zero-extend) */
	    "rd_sp rs1_sp !16 addi"
	    "rs1_t0 @_read_end bnez"        /* if bytes != 0, return byte */
	    "rd_a0 !-1 addi"                /* else: a0 = -1 (EOF) */
	    ":_read_end");
}

/* _write -- EFI_FILE_PROTOCOL.Write(This, BufferSize*, Buffer*).
 * Single-byte write (we only ever pass size=1 from fputc). */
long _write(FILE* f, unsigned size, char c, FUNCTION write)
{
	/* fout->write(fout, &size, &c) */
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rd_t0 rs1_fp !-16 ld"
	    "rs1_sp rs2_t0 @8 sd"
	    "rd_t0 rs1_fp !-24 ld"
	    "rs1_sp rs2_t0 sd"
	    "rd_a1 rs1_sp !8 addi"
	    "rd_a2 rs1_sp mv"
	    "rd_t0 rs1_fp !-32 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !32 addi");
}

/* _write_stdout -- EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL.OutputString.
 * UEFI's text console expects a null-terminated UCS-2 string, but
 * we only need to print one byte: pad to two bytes (high byte = 0)
 * with a trailing UCS-2 null and call OutputString. */
void _write_stdout(void* con_out, int c, FUNCTION output_string)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rd_t0 rs1_fp !-16 ld"
	    "rs1_sp rs2_t0 sd"
	    "rs1_sp rs2_zero @8 sd"
	    "rd_a1 rs1_sp mv"
	    "rd_t0 rs1_fp !-24 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !32 addi");
}

/* _open_protocol -- EFI_BOOT_SERVICES.OpenProtocol.
 * Returns the protocol-specific interface pointer (e.g.,
 * EFI_LOADED_IMAGE_PROTOCOL*) by writing through an output-pointer
 * argument; the asm reserves a stack slot for that, then loads it
 * into a0 after the UEFI call. */
void* _open_protocol(void* handle, struct efi_guid* protocol, void* agent_handle, void* controller_handle, long attributes, FUNCTION open_protocol)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_a1 rs1_fp !-16 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_zero sd"
	    "rd_a2 rs1_sp mv"
	    "rd_a3 rs1_fp !-24 ld"
	    "rd_a4 rs1_fp !-32 ld"
	    "rd_a5 rs1_fp !-40 ld"
	    "rd_t0 rs1_fp !-48 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi"
	    "rd_a0 rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* _close_protocol -- EFI_BOOT_SERVICES.CloseProtocol.
 * No output; just dispatches the call. */
int _close_protocol(void *handle, struct efi_guid* protocol, void* agent_handle, void* controller_handle, FUNCTION close_protocol)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_a1 rs1_fp !-16 ld"
	    "rd_a2 rs1_fp !-24 ld"
	    "rd_a3 rs1_fp !-32 ld"
	    "rd_t0 rs1_fp !-40 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* _open_volume -- EFI_SIMPLE_FILE_SYSTEM_PROTOCOL.OpenVolume.
 * Returns the root-directory EFI_FILE_PROTOCOL* via an output ptr. */
int _open_volume(struct efi_simple_file_system_protocol* rootfs, FUNCTION open_volume)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_zero sd"
	    "rd_a1 rs1_sp mv"
	    "rd_t0 rs1_fp !-16 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi"
	    "rd_a0 rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* _open -- EFI_FILE_PROTOCOL.Open.
 * Opens a file relative to a directory handle (here `_rootdir`).
 * Returns the new EFI_FILE_PROTOCOL* via an output ptr. */
FILE* _open(void* _rootdir, char* name, long mode, long attributes, FUNCTION open)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_zero sd"
	    "rd_a1 rs1_sp mv"
	    "rd_a2 rs1_fp !-16 ld"
	    "rd_a3 rs1_fp !-24 ld"
	    "rd_a4 rs1_fp !-32 ld"
	    "rd_t0 rs1_fp !-40 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi"
	    "rd_a0 rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* _close -- EFI_FILE_PROTOCOL.Close. UEFI flushes & releases the
 * handle; we don't bother reading the status. */
FILE* _close(FILE* f, FUNCTION close)
{
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_t0 rs1_fp !-16 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* _allocate_pages -- EFI_BOOT_SERVICES.AllocatePages.
 * UEFI signature is allocate_pages(type, memory_type, pages,
 * EFI_PHYSICAL_ADDRESS*). The fourth arg is BOTH input and output:
 * input is the requested base when type=ADDRESS, output is the
 * actual allocation address. We pass &_malloc_ptr (a stack-frame
 * slot at fp-32), then re-load that slot into a0 to return the
 * allocated address rather than UEFI's status code (the caller
 * needs the address; UEFI returns 0 on success which would be
 * indistinguishable from a NULL result).
 *
 * Returns _malloc_ptr and not exit value from AllocatePages call. */
long _allocate_pages(unsigned type, unsigned memory_type, unsigned pages, long _malloc_ptr, FUNCTION allocate_pages)
{
	/* boot->allocate_pages(type, memory_type, pages, &_malloc_ptr) */
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_a1 rs1_fp !-16 ld"
	    "rd_a2 rs1_fp !-24 ld"
	    "rd_a3 rs1_fp !-32 addi"
	    "rd_t0 rs1_fp !-40 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi"
	    "rd_a0 rs1_fp !-32 ld");
}

/* _free_pages -- EFI_BOOT_SERVICES.FreePages. */
void _free_pages(void* memory, unsigned pages, FUNCTION free_pages)
{
	/* boot->free_pages(memory, pages) */
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_a1 rs1_fp !-16 ld"
	    "rd_t0 rs1_fp !-24 ld"
	    "rd_sp rs1_sp !-16 addi"
	    "rs1_sp rs2_ra sd"
	    "rd_ra rs1_t0 jalr"
	    "rd_ra rs1_sp ld"
	    "rd_sp rs1_sp !16 addi");
}

/* === Section 5: Higher-level libc surface ===========================
 * Standard fgetc/fread/fputc/fwrite/fputs/fopen/fclose/strlen on top
 * of the asm wrappers. These are plain C -- no inline asm -- so
 * cc_riscv64 emits ordinary RISC-V code for them. */

/* fgetc reads exactly one byte from f via UEFI's File.Read. The
 * underlying _read returns -1 (EOF) when 0 bytes were actually
 * read, which sign-extends to 0xFFFFFFFF when promoted. */
int fgetc(FILE* f)
{
	struct efi_file_protocol* file = f;

	unsigned size = 1;
	char c = _read(file, size, file->read);
	return c;
}

unsigned fread(char* buffer, unsigned size, unsigned count, FILE* f) {
	count = size * count;

	unsigned i = 0;
	for(; i < count; i = i + 1) {
		buffer[i] = fgetc(f);
	}

	return i;
}

/* fputc -- write one byte. If f is the stdout/stderr sentinel value
 * (not a real EFI_FILE_PROTOCOL pointer), route through ConOut and
 * also emit '\r' on '\n' so the UEFI text console doesn't drift the
 * cursor when it sees a bare LF. Otherwise dispatch to File.Write. */
void fputc(char c, FILE* f)
{
	unsigned size = 1;
	/* In UEFI StdErr might not be printing stuff to console, so just use stdout */
	if(f == stdout || f == stderr)
	{
		_write_stdout(_system->con_out, c, _system->con_out->output_string);
		if('\n' == c)
		{
			_write_stdout(_system->con_out, '\r', _system->con_out->output_string);
		}
		return;
	}
	struct efi_file_protocol* file = f;
	_write(file, size, c, file->write);
}

unsigned fwrite(char* buffer, unsigned size, unsigned count, FILE* f) {
	count = size * count;

	unsigned i = 0;
	for(; i < count; i = i + 1) {
		fputc(buffer[i], f);
	}

	return i;
}

void fputs(char* s, FILE* f)
{
	while(0 != s[0])
	{
		fputc(s[0], f);
		s = s + 1;
	}
}

int strlen(char* str)
{
	int i = 0;
	while(0 != str[i]) i = i + 1;
	return i;
}

char* _posix_path_to_uefi(char *narrow_string);

/* fopen -- POSIX-style fopen on top of UEFI's File.Open.
 *   * Path conversion: '/' -> '\\', narrow ASCII -> UCS-2.
 *   * Mode: 'w' -> CREATE | READ | WRITE (the magic 1<<63 bit is
 *     EFI_FILE_MODE_CREATE; the riscv64 build needs the literal
 *     because cc_riscv64 doesn't accept hex constants >= 1<<31).
 *     Anything else (including 'r') opens read-only. */
FILE* fopen(char* filename, char* mode)
{
	char* wide_filename = _posix_path_to_uefi(filename);
	FILE* f;
	long status;
	if('w' == mode[0])
	{
		long mode = 1 << 63; /* EFI_FILE_MODE_CREATE = 0x8000000000000000 */
		mode = mode | EFI_FILE_MODE_WRITE | EFI_FILE_MODE_READ;
		f = _open(_rootdir, wide_filename, mode, 0, _rootdir->open);
	}
	else
	{       /* Everything else is a read */
		f = _open(_rootdir, wide_filename, EFI_FILE_MODE_READ, EFI_FILE_READ_ONLY, _rootdir->open);
	}
	return f;
}

int fclose(FILE* stream)
{
	struct efi_file_protocol* file = stream;
	return _close(file, file->close);
}

/* === Section 6: Memory management ===================================
 * Bump-pointer allocator on a single AllocatePages slab. No free()
 * (we just leak; UEFI reclaims everything when the image exits).
 * On first malloc we ask UEFI for PAGE_NUM x PAGE_SIZE bytes once;
 * subsequent mallocs walk _malloc_ptr forward until _brk_ptr. */

/* A very primitive memory manager */
void* malloc(int size)
{
	if(NULL == _brk_ptr)
	{
		unsigned pages = PAGE_NUM; /* 64 MiB = 16384 * 4 KiB pages */
		_malloc_ptr = _allocate_pages(EFI_ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, pages, _malloc_ptr, _system->boot_services->allocate_pages);
		if(_malloc_ptr == 0)
		{
			return 0;
		}
		_brk_ptr = _malloc_ptr + (pages * PAGE_SIZE);
	}

	/* We never allocate more memory in bootstrap mode */
	if(_brk_ptr < _malloc_ptr + size)
	{
		return 0;
	}

	long old_malloc = _malloc_ptr;
	_malloc_ptr = _malloc_ptr + size;
	return old_malloc;
}

void* memset(void* ptr, int value, int num)
{
	char* s;
	for(s = ptr; 0 < num; num = num - 1)
	{
		s[0] = value;
		s = s + 1;
	}
}

void* calloc(int count, int size)
{
	void* ret = malloc(count * size);
	if(NULL == ret) return NULL;
	memset(ret, 0, (count * size));
	return ret;
}

void free(void* l)
{
	return;
}

/* exit -- branch into libc-{core,full}.M1's :FUNCTION__exit, which
 * runs _cleanup, restores UEFI's sp, pops callee-saved regs, and
 * `ret`s back to UEFI. The `goto` is a cc_riscv64 idiom for jumping
 * to a label that lives outside any C function (M1-defined). */
void exit(unsigned value)
{
	goto FUNCTION__exit;
}

/* === Section 7: UEFI <-> POSIX glue ===================================
 * UEFI uses Windows-style paths ('\\') and UCS-2 (16-bit) strings.
 * POSIX programs and our C code use '/' and ASCII. These helpers
 * convert both directions inline. */

/* _posix_path_to_uefi -- narrow ASCII path -> UCS-2 with '\\'.
 * Caller-allocated output via calloc; output is NUL-terminated as a
 * UCS-2 string (i.e. trailing 0x00 0x00). */
void _posix_path_to_uefi(char *narrow_string)
{
	unsigned length = strlen(narrow_string) + 1;
	char *wide_string = calloc(length, 2);
	unsigned i;
	for(i = 0; i < length; i = i + 1)
	{
		if(narrow_string[i] == '/')
		{
			wide_string[2 * i] = '\\';
		}
		else
		{
			wide_string[2 * i] = narrow_string[i];
		}
	}
	return wide_string;
}

char* wide2string(char *wide_string, unsigned length)
{
	/* Scan for UCS-2 null terminator (two consecutive zero bytes). The passed
	 * `length` from image->load_options_size is unreliable under cc_riscv64
	 * (it reads UINT32 struct fields as 8 bytes, picking up garbage padding).
	 * Cap scan at 4096 to avoid runaway reads if the buffer has no null. */
	unsigned i;
	unsigned real_len = 0;
	unsigned cap = 4096;
	while(real_len < cap)
	{
		if(wide_string[2 * real_len] == 0)
		{
			if(wide_string[2 * real_len + 1] == 0) break;
		}
		real_len = real_len + 1;
	}
	char *narrow_string = calloc(real_len + 1, 1);
	for(i = 0; i < real_len; i = i + 1)
	{
		narrow_string[i] = wide_string[2 * i];
	}
	return narrow_string;
}

int is_space(char c)
{
	return (c == ' ') || (c == '\t');
}

/* process_load_options -- split the LoadOptions string into argc/argv.
 *
 * UEFI's LOADED_IMAGE_PROTOCOL.LoadOptions arrives as a single
 * UCS-2 string (already converted to ASCII by wide2string) such as
 *     "myprog --flag a.txt b.txt"
 * with spaces/tabs separating tokens. We:
 *   1. Walk the string to count whitespace transitions -> argc.
 *   2. calloc argv[argc+1].
 *   3. Walk again, NUL-terminating each token in-place and storing
 *      the start of each token into argv[j]. */
void process_load_options(char* load_options)
{
	/* Determine argc */
	_argc = 1; /* command name */
	char *i = load_options;
	unsigned was_space = 0;
	do
	{
		if(is_space(i[0]))
		{
			if(!was_space)
			{
				_argc = _argc + 1;
				was_space = 1;
			}
		}
		else
		{
			was_space = 0;
		}
		i = i + 1;
	} while(i[0] != 0);

	/* Collect argv */
	_argv = calloc(_argc + 1, sizeof(char*));
	i = load_options;
	unsigned j;
	for(j = 0; j < _argc; j = j + 1)
	{
		_argv[j] = i;
		do
		{
			i = i + 1;
		} while(!is_space(i[0]) && i[0] != 0);
		i[0] = 0;
		do
		{
			i = i + 1;
		} while(is_space(i[0]));
	}
}

long _clamp_u32(long x)
{
	/* Zero-extend low 32 bits via slli+srli. cc_riscv64 reads UINT32 struct
	 * fields as 8 bytes (ld), picking up 4 bytes of garbage padding as the
	 * upper half. This helper masks to just the low 32 bits. */
	asm("rd_a0 rs1_fp !-8 ld"
	    "rd_a0 rs1_a0 !32 slli"
	    "rd_a0 rs1_a0 !32 srli");
}

/* === Section 8: _init / _cleanup ====================================
 * Called by libc-{core,full}.M1's _start. _init opens the FS root
 * (via SIMPLE_FILE_SYSTEM_PROTOCOL on the LOADED_IMAGE's device),
 * parses LoadOptions into argv, and reserves the user stack.
 * _cleanup unwinds in reverse before the image returns to UEFI. */

/* _init -- UEFI runtime initialization.
 * Steps:
 *   1. Allocate USER_STACK_SIZE bytes for the larger user stack
 *      that libc-core.M1's exit_uefi_stack will switch to.
 *   2. Build EFI_LOADED_IMAGE_PROTOCOL_GUID at runtime -- UEFI GUIDs
 *      live in PE32 .data sections normally, but cc_riscv64 only
 *      supports global constants up to 31 bits, so we assemble each
 *      half from two 31-bit values plus an overflow offset.
 *   3. Open the LOADED_IMAGE protocol on _image_handle, read its
 *      LoadOptions, parse argc/argv via process_load_options.
 *   4. Build EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, open it on the
 *      image's device, and OpenVolume -> _rootdir. */
void _init()
{
	/* Allocate user stack, UEFI stack is not big enough for compilers */
	__user_stack = malloc(USER_STACK_SIZE);
	_malloc_start = __user_stack;
	/* Go to the other end of allocated memory, as stack grows downwards)*/
	__user_stack = __user_stack + USER_STACK_SIZE;

	/* Process command line arguments */
	EFI_LOADED_IMAGE_PROTOCOL_GUID = calloc(1, sizeof(struct efi_guid));
	EFI_LOADED_IMAGE_PROTOCOL_GUID->data1 = (0x11D29562 << 32) + 0x5B1B31A1;
	/* We want to add 0xA0003F8E but M2 treats 32-bit values as negatives, in order to
	 * have the same behaviour on 32-bit systems, so restrict to 31-bit constants */
	EFI_LOADED_IMAGE_PROTOCOL_GUID->data2 = (0x3B7269C9 << 32) + 0x50003F8E + 0x50000000;

	struct efi_loaded_image_protocol* image = _open_protocol(_image_handle, EFI_LOADED_IMAGE_PROTOCOL_GUID, _image_handle, 0, EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL, _system->boot_services->open_protocol);
	/* load_options_size is UEFI UINT32 (4 bytes + 4 padding). cc_riscv64 emits
	 * 8-byte ld for all struct fields, so it picks up garbage padding bytes
	 * as the upper 32 bits. Mask to 32 bits via _clamp_u32. */
	unsigned los = _clamp_u32(image->load_options_size);
	char* load_options = wide2string(image->load_options, los);
	process_load_options(load_options);

	EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID = calloc(1, sizeof(struct efi_guid));
	EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID->data1 = (0x11D26459 << 32) + 0x564E5B22 + 0x40000000;
	EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID->data2 = (0x3B7269C9 << 32) + 0x5000398E + 0x50000000;

	_root_device = image->device;
	struct efi_simple_file_system_protocol* rootfs = _open_protocol(_root_device, EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, _image_handle, 0, EFI_OPEN_PROTOCOL_BY_HANDLE_PROTOCOL, _system->boot_services->open_protocol);
	_rootdir = _open_volume(rootfs, rootfs->open_volume);
}

/* _cleanup -- release UEFI resources before image exit.
 * Unwinds _init in reverse order: close root dir handle, close the
 * SIMPLE_FILE_SYSTEM and LOADED_IMAGE protocols, free the malloc
 * page slab. UEFI would clean these up automatically when the
 * image exits, but releasing explicitly avoids the warning some
 * firmwares print about unreleased protocols. */
void _cleanup()
{
	fclose(_rootdir);
	_close_protocol(_root_device, EFI_SIMPLE_FILE_SYSTEM_PROTOCOL_GUID, _image_handle, 0, _system->boot_services->close_protocol);
	_close_protocol(_image_handle, EFI_LOADED_IMAGE_PROTOCOL_GUID, _image_handle, 0, _system->boot_services->close_protocol);
	_free_pages(_malloc_start, PAGE_NUM, _system->boot_services->free_pages);
}
