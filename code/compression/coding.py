"""
This file implements coding methdos for Miracle

Currently implemented:
 - Arithmetic coding

"""

import numpy as np

# ==============================================================================
# Helper Functions
# ==============================================================================

def write_bin_code(code, path):

    # Pad the code
    code += "0" * (8 - len(code) % 8) if len(code) % 8 != 0 else ""

    message_bytes = [int('0b' + code[s:s + 8], 2) for s in range(0, len(code), 8)]

    with open(path, "wb") as compressed_file:
        compressed_file.write(bytes(message_bytes))


def read_bin_code(path):

    with open(path, "rb") as compressed_file:
        compressed = ''.join(["{:08b}".format(x) for x in compressed_file.read()])

    return compressed

# ==============================================================================
# Arithmetic coding
# ==============================================================================

class ArithmeticCoder(object):

    def __init__(self, P, precision=32):

        self._P = P
        self._precision = precision


        # Calculates the (unnormalized) CDF from P as well as its total mass
        C = []
        D = []

        c = 0

        for p in P:

            C.append(c)

            c += p

            D.append(c)

        self.C = C
        self.D = D
        self.R = D[-1]

    # ---------------------------------------------------------------------------

    def encode(self, message):

        precision = self._precision

        # Calculate some stuff
        C, D, R = self.C, self.D, self.R

        whole = 2**precision
        half = 2**(precision - 1)
        quarter = 2**(precision - 2)

        low = 0
        high = whole
        s = 0

        code = ""

        for k in range(len(message)):

            width = high - low

            # Find interval for next symbol
            high = low + (width * D[message[k]]) // R
            low = low + (width * C[message[k]]) // R

            # Interval subdivision
            while high < half or low > half:

                # First case: we're in the lower half
                if high < half:
                    code += "0" + "1" * s
                    s = 0

                    # Interval rescaling
                    low *= 2
                    high *= 2

                # Second case: we're in the upper half
                elif low > half:
                    code += "1" + "0" * s
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
            code += "0" + "1" * s
        else:
            code += "1" + "0" * s

        return code

    # ------------------------------------------------------------------------------

    def decode(self, code):

        precision = self._precision

        # Calculate some stuff
        C, D, R = self.C, self.D, self.R

        whole = 2**precision
        half = 2**(precision - 1)
        quarter = 2**(precision - 2)

        low = 0
        high = whole

        # Initialize representation of binary lower bound
        z = 0
        i = 0

        while i < precision and i < len(code):
            if code[i] == '1':
                z += 2**(precision - i - 1)
            i += 1

        message = []


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
                        return message

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
