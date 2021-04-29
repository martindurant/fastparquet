# https://cython.readthedocs.io/en/latest/src/userguide/
#   source_files_and_compilation.html#compiler-directives
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# cython: language_level=3
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: cdivision=True
# cython: always_allow_keywords=False

import cython
cdef extern from "string.h":
    void *memcpy(void *dest, const void *src, size_t n)
from cpython cimport PyBytes_FromStringAndSize


cpdef void read_rle(NumpyIO file_obj, int header, int bit_width, NumpyIO o):
    """Read a run-length encoded run from the given fo with the given header and bit_width.

    The count is determined from the header and the width is used to grab the
    value that's repeated. Yields the value repeated count times.
    """
    cdef int count, width, i, data = 0
    count = header >> 1
    width = (bit_width + 7) // 8
    for i in range(width):
        data <<= 8
        data |= (<int>file_obj.read_byte()) & 0xff
    for i in range(count):
        o.write_int(data)


cpdef int width_from_max_int(long value):
    """Convert the value specified to a bit_width."""
    cdef int i
    for i in range(0, 64):
        if value == 0:
            return i
        value >>= 1


cdef int _mask_for_bits(int i):
    """Generate a mask to grab `i` bits from an int value."""
    return (1 << i) - 1


cpdef void read_bitpacked(NumpyIO file_obj, int header, int width, NumpyIO o):
    """
    Read values packed into width-bits each (which can be >8)
    """
    cdef unsigned int count, mask, data
    cdef unsigned char left = 8, right = 0
    # TODO: special case for width=4, 8

    count = ((header & 0xff) >> 1) * 8
    mask = _mask_for_bits(width)
    data = 0xff & <int>file_obj.read_byte()
    while count:
        if right > 8:
            data >>= 8
            left -= 8
            right -= 8
        elif left - right < width:
            data |= (file_obj.read_byte() & 0xff) << left
            left += 8
        else:
            o.write_int(<int>(data >> right & mask))
            count -= 1
            right += width


cpdef unsigned long read_unsigned_var_int(NumpyIO file_obj):
    """Read a value using the unsigned, variable int encoding.
    file-obj is a NumpyIO of bytes; avoids struct to allow numba-jit
    """
    cdef unsigned long result = 0
    cdef int shift = 0
    cdef char byte
    while True:
        byte = file_obj.read_byte()
        result |= (<long>(byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result


cpdef void read_rle_bit_packed_hybrid(NumpyIO io_obj, int width, int length, NumpyIO o):
    """Read values from `io_obj` using the rel/bit-packed hybrid encoding.

    If length is not specified, then a 32-bit int is read first to grab the
    length of the encoded data.

    file-obj is a NumpyIO of bytes; o if an output NumpyIO of int32

    The caller can tell the number of elements in the output by looking
    at .tell().
    """
    cdef int start, header
    if length is False:
        length = io_obj.read_int()
    start = io_obj.tell()
    while io_obj.tell() - start < length and o.tell() < o.nbytes:
        header = <int>read_unsigned_var_int(io_obj)
        if header & 1 == 0:
            read_rle(io_obj, header, width, o)
        else:
            read_bitpacked(io_obj, header, width, o)


cdef void encode_unsigned_varint(int x, NumpyIO o):  # pragma: no cover
    while x > 127:
        o.write_byte((x & 0x7F) | 0x80)
        x >>= 7
    o.write_byte(x)


@cython.wraparound(False)
@cython.boundscheck(False)
def encode_bitpacked(int[:] values, int width, NumpyIO o):  # pragma: no cover
    """
    Write values packed into width-bits each (which can be >8)

    values is a NumbaIO array (int32)
    o is a NumbaIO output array (uint8), size=(len(values)*width)/8, rounded up.
    """

    cdef int bit_packed_count = (values.shape[0] + 7) // 8
    encode_unsigned_varint(bit_packed_count << 1 | 1, o)  # write run header
    cdef char right_byte_mask = 0b11111111
    cdef int bit=0, bits=0, v, counter
    for counter in range(values.shape[0]):
        v = values[counter]
        bits |= v << bit
        bit += width
        while bit >= 8:
            o.write_byte(bits & right_byte_mask)
            bit -= 8
            bits >>= 8
    if bit:
        o.write_byte(bits)



cdef class NumpyIO(object):
    """
    Read or write from a numpy array like a file object

    The main purpose is to keep track of the current location in the memory
    """
    cdef char[:] data
    cdef unsigned int loc, nbytes
    cdef char* ptr

    def __init__(self, char[:] data):
        self.data = data
        self.loc = 0
        self.ptr = &data[0]
        self.nbytes = data.shape[0]

    cdef char* get_pointer(self):
        return self.ptr + self.loc

    @property
    def len(self):
        return self.nbytes

    @cython.wraparound(False)
    cpdef char[:] read(self, int x):
        cdef char[:] out
        out = self.data[self.loc:self.loc + x]
        self.loc += x
        return out

    cpdef char read_byte(self):
        cdef char out
        out = self.ptr[self.loc]
        self.loc += 1
        return out

    cpdef int read_int(self):
        cdef int i
        if self.nbytes - self.loc < 4:
            return 0
        i = (<int*> self.get_pointer())[0]
        self.loc += 4
        return i

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void write(self, char[:] d):
        for i in range(d.shape[0]):
            self.write_byte(d[i])

    cdef void write_byte(self, char b):
        if self.loc >= self.nbytes:
            # ignore attempt to write past end of buffer
            return
        self.ptr[self.loc] = b
        self.loc += 1

    cpdef void write_int(self, int i):
        if self.nbytes - self.loc < 4:
            return
        (<int*> self.get_pointer())[0] = i
        self.loc += 4

    cdef void write_many(self, char b, int count):
        cdef int i
        for i in range(count):
            self.write_byte(b)

    cpdef int tell(self):
        return self.loc

    cpdef void seek(self, int loc, int whence=0):
        if whence == 0:
            self.loc = loc
        elif whence == 1:
            self.loc += loc
        elif whence == 2:
            self.loc = self.nbytes + loc
        if self.loc > self.nbytes:
            self.loc = self.nbytes

    @cython.wraparound(False)
    cpdef char[:] so_far(self):
        """ In write mode, the data we have gathered until now
        """
        return self.data[:self.loc]


@cython.wraparound(False)
@cython.boundscheck(False)
def _assemble_objects(object[:] assign, int[:] defi, int[:] rep, val, dic, d,
                      char null, null_val, int max_defi, int prev_i):
    """Dremel-assembly of arrays of values into lists

    Parameters
    ----------
    assign: array dtype O
        To insert lists into
    defi: int array
        Definition levels, max 3
    rep: int array
        Repetition levels, max 1
    dic: array of labels or None
        Applied if d is True
    d: bool
        Whether to dereference dict values
    null: bool
        Can an entry be None?
    null_val: bool
        can list elements be None
    max_defi: int
        value of definition level that corresponds to non-null
    prev_i: int
        1 + index where the last row in the previous page was inserted (0 if first page)
    """
    cdef int counter, i, re, de
    cdef int vali = 0
    cdef char started = False, have_null = False
    if d:
        # dereference dict values
        val = dic[val]
    i = prev_i
    part = []
    for counter in range(rep.shape[0]):
        de = defi[counter] if defi is not None else max_defi
        re = rep[counter]
        if not re:
            # new row - save what we have
            if started:
                assign[i] = None if have_null else part
                part = []
                i += 1
            else:
                # first time: no row to save yet, unless it's a row continued from previous page
                if vali > 0:
                    assign[i - 1].extend(part) # add the items to previous row
                    part = []
                    # don't increment i since we only filled i-1
                started = True
        if de == max_defi:
            # append real value to current item
            part.append(val[vali])
            vali += 1
        elif de > null:
            # append null to current item
            part.append(None)
        # next object is None as opposed to an object
        have_null = de == 0 and null
    if started: # normal case - add the leftovers to the next row
        assign[i] = None if have_null else part
    else: # can only happen if the only elements in this page are the continuation of the last row from previous page
        assign[i - 1].extend(part)
    return i


cdef int zigzag_int(unsigned long n):
    return (n >> 1) ^ -(n & 1)


cdef long zigzag_long(unsigned long n):
    return (n >> 1) ^ -(n & 1)


cpdef dict read_thrift(NumpyIO data):
    cdef char byte, id = 0, bit
    cdef int size
    out = {}
    while True:
        byte = data.read_byte()
        if byte == 0:
            break
        id += (byte & 0b11110000) >> 4
        bit = byte & 0b00001111
        if bit == 1:
            out[id] = True
        elif bit == 2:
            out[id] = False
        elif bit == 5 or bit == 6:
            out[id] = zigzag_long(read_unsigned_var_int(data))
        elif bit == 7:
            out[id] = <double>data.get_pointer()[0]
            data.seek(4, 1)
        elif bit == 8:
            size = read_unsigned_var_int(data)
            out[id] = PyBytes_FromStringAndSize(data.get_pointer(), size)
            data.seek(size, 1)
        elif bit == 9:
            out[id] = read_list(data)
        elif bit == 12:
            out[id] = read_thrift(data)
    return out


cdef list read_list(NumpyIO data):
    cdef char byte, typ
    cdef int size, bsize, _
    byte = data.read_byte()
    if byte >= 0xf0:  # 0b11110000
        size = read_unsigned_var_int(data)
    else:
        size = ((byte & 0xf0) >> 4)
    out = []
    typ = byte & 0x0f # 0b00001111
    if typ == 5:
        for _ in range(size):
            out.append(zigzag_int(read_unsigned_var_int(data)))
    elif typ == 8:
        for _ in range(size):
            bsize = read_unsigned_var_int(data)
            out.append(PyBytes_FromStringAndSize(data.get_pointer(), size))
            data.seek(bsize, 1)
    else:
        for _ in range(size):
            out.append(read_thrift(data))

    return out