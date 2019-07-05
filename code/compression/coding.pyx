"""
This file implements coding methdos for Miracle

Currently implemented:
 - Arithmetic coding

"""

from collections import deque
from data_structures import IntervalAVLTree

import numpy as np
cimport numpy as np

from tqdm import tqdm

DTYPE = np.int64

ctypedef np.int64_t DTYPE_t


# ==============================================================================
# Helper Functions
# ==============================================================================

def write_bin_code(code, path, extras=None):

    # Pad the code
    code += "0" * (8 - len(code) % 8) if len(code) % 8 != 0 else ""

    message_bytes = [int('0b' + code[s:s + 8], 2) for s in range(0, len(code), 8)]

    with open(path, "wb") as compressed_file:
        
        if extras is not None:
            for extra in extras:
                compressed_file.write(bytes([extra // 256, extra % 256]))
        
        compressed_file.write(bytes(message_bytes))


def read_bin_code(path, num_extras=0):

    with open(path, "rb") as compressed_file:
        compressed = ''.join(["{:08b}".format(x) for x in compressed_file.read()])

    extra_bits = compressed[:num_extras * 16]
    compressed = compressed[num_extras * 16:]
    
    extras = [int('0b' + extra_bits[s:s + 16], 2) for s in range(0, num_extras * 16, 16)]
    
    return compressed, extras

# ==============================================================================
# Arithmetic coding
# ==============================================================================

class ArithmeticCoder(object):

    def __init__(self, np.ndarray P, precision=32):

        self._P = P
        self._precision = precision


        # Calculates the (unnormalized) CDF from P as well as its total mass
        cdef np.ndarray C = np.zeros_like(P, dtype=DTYPE)
        cdef np.ndarray D = np.zeros_like(P, dtype=DTYPE)

        cdef long c = 0
        cdef int i

        for i in range(len(P)):

            C[i] = c

            c += P[i]

            D[i] = c

        self.symbol_tree = IntervalAVLTree(C)
        print("Depth of symbol tree: {}".format(self.symbol_tree.depth))

        self.C = np.array(C)
        self.D = np.array(D)
        self.R = D[-1]

    # ---------------------------------------------------------------------------

    def encode(self, message):

        cdef long precision = self._precision

        # Calculate some stuff
        cdef np.ndarray C = self.C
        cdef np.ndarray D = self.D
        cdef long R = self.R


        cdef long whole = 2**precision
        cdef long half = 2**(precision - 1)
        cdef long quarter = 2**(precision - 2)

        cdef long low = 0
        cdef long high = whole
        cdef int s = 0

        cdef long low_, high_

        code = deque([])

        cdef int k

        for k in tqdm(range(len(message))):

            width = high - low

            # Find interval for next symbol
            high = low + (width * D[message[k]]) // R
            low = low + (width * C[message[k]]) // R

            # Interval subdivision
            while high < half or low > half:

                # First case: we're in the lower half
                if high < half:
                    code.extend("0" + "1" * s)
                    s = 0

                    # Interval rescaling
                    low *= 2
                    high *= 2

                # Second case: we're in the upper half
                elif low > half:
                    code.extend("1" + "0" * s)
                    s = 0

                    low = (low - half) * 2
                    high = (high - half) * 2

            # Middle rescaling
            while low > quarter and high < 3 * quarter:
                s += 1
                low = (low - quarter) * 2
                high = (high - quarter) * 2

        # Final emission step
        s += 1

        if low <= quarter:
            code.extend("0" + "1" * s)
        else:
            code.extend("1" + "0" * s)

        return list(code)

    # ------------------------------------------------------------------------------

    def decode(self, code):

        cdef long precision = self._precision

        # Calculate some stuff
        cdef np.ndarray C = self.C
        cdef np.ndarray D = self.D
        cdef long R = self.R

        cdef long whole = 2**precision
        cdef long half = 2**(precision - 1)
        cdef long quarter = 2**(precision - 2)

        cdef long low = 0
        cdef long high = whole

        cdef long low_, high_, width

        # Initialize representation of binary lower bound
        cdef long z = 0
        cdef long i = 0
        cdef int j

        while i < precision and i < len(code):
            if code[i] == '1':
                z += 2**(precision - i - 1)
            i += 1

        message = deque([])

        while True:

            # Find the current symbol
            for j in range(len(C)):

                width = high - low

                # Find interval for next symbol
                high_ = low + (width * D[j]) // R
                low_ = low + (width * C[j]) // R

                if low_ <= z < high_:

                    # Emit the current symbol
                    message.append(j)

                    # Update bounds
                    high = high_
                    low = low_

                    # Are we at the end?
                    if j == 0:
                        return list(message)

                    # Interval rescaling
                    while high < half or low > half:

                        # First case: we're in the lower half
                        if high < half:
                            low *= 2
                            high *= 2

                            z *= 2

                        # Second case: we're in the upper half
                        elif low > half:
                            low = (low - half) * 2
                            high = (high - half) * 2

                            z = (z - half) * 2

                        # Update the precision of the lower bound
                        if i < len(code) and code[i] == '1':
                            z += 1

                        i += 1

                    # Middle rescaling
                    while low > quarter and high < 3 * quarter:
                        low = (low - quarter) * 2
                        high = (high - quarter) * 2
                        z = (z - quarter) * 2

                        # Update the precision of the lower bound
                        if i < len(code) and code[i] == '1':
                            z += 1

                        i += 1

    # ------------------------------------------------------------------------------

    def decode_fast(self, code):

        cdef long precision = self._precision

        # Calculate some stuff
        cdef np.ndarray C = self.C
        cdef np.ndarray D = self.D
        cdef long R = self.R

        cdef long whole = 2**precision
        cdef long half = 2**(precision - 1)
        cdef long quarter = 2**(precision - 2)

        cdef long low = 0
        cdef long high = whole

        cdef long low_, high_, width

        # Initialize representation of binary lower bound
        cdef long z = 0
        cdef long i = 0
        cdef int j

        while i < precision and i < len(code):
            if code[i] == '1':
                z += 2**(precision - i - 1)
            i += 1

        message = deque([])

        while True:

            width = high - low

            # Find the current symbol
            transformer = lambda x: (width * x) // R
            lower_bound,  j = self.symbol_tree.find_tightest_lower_bound(z - low,
                                                                         transformer=transformer)

            # Find interval for next symbol
            low_ = low + lower_bound
            high_ = low + (width * D[j]) // R

            # Emit the current symbol
            message.append(j)

            # Update bounds
            high = high_
            low = low_

            # Are we at the end?
            if j == 0:
                return list(message)

            # Interval rescaling
            while high < half or low > half:

                # First case: we're in the lower half
                if high < half:
                    low *= 2
                    high *= 2

                    z *= 2

                # Second case: we're in the upper half
                elif low > half:
                    low = (low - half) * 2
                    high = (high - half) * 2

                    z = (z - half) * 2

                # Update the precision of the lower bound
                if i < len(code) and code[i] == '1':
                    z += 1

                i += 1

            # Middle rescaling
            while low > quarter and high < 3 * quarter:
                low = (low - quarter) * 2
                high = (high - quarter) * 2
                z = (z - quarter) * 2

                # Update the precision of the lower bound
                if i < len(code) and code[i] == '1':
                    z += 1

                i += 1

# ==============================================================================
# TODO: ANS coding
# ==============================================================================

