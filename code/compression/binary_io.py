import numpy as np

# ==============================================================================
# Helper Functions
# ==============================================================================

def to_bit_string(num, num_bits):
    
    if num >= 2**num_bits:
        raise Exception("The number is bigger than what we can encode!")
        
    bitcode = []
    
    for i in range(num_bits):
        bitcode.append(str(num % 2))
        
        num //= 2
        
    return ''.join(bitcode)

def from_bit_string(bitcode):
    
    num = 0
    
    for i in range(len(bitcode)):
        
        if bitcode[i] == "1":
            num += 2**i
        
    return num

def write_bin_code(code, path, extras=None, var_length_extras=None, var_length_bytes=None):
    
    if var_length_extras is not None:
        if var_length_bytes is None or len(var_length_extras) != len(var_length_bytes):
            raise Exception("Each var length extra needs to have a bytelength associated!")
            
    # Pad the code
    code += "0" * (8 - len(code) % 8) if len(code) % 8 != 0 else ""

    message_bytes = [int('0b' + code[s:s + 8], 2) for s in range(0, len(code), 8)]

    with open(path, "wb") as compressed_file:
        
        if extras is not None:
            for extra in extras:
                compressed_file.write(bytes([extra // 256, extra % 256]))
                
        if var_length_extras is not None:
            for extra, extra_byte_size in zip(var_length_extras, var_length_bytes):
                
                # First code the length of the extra on 16 bits
                compressed_file.write(bytes([len(extra) // 256, len(extra) % 256]))
                
                # Write the rest on var_length_bytes bytes
                for item in extra:
                    
                    byte_list = []
                    for i in range(extra_byte_size - 1, -1, -1):
                        octet = int(256**i)
                        
                        byte_list.append(item // octet)
                        item = item % octet

                    compressed_file.write(bytes(byte_list))
                
        
        compressed_file.write(bytes(message_bytes))


def read_bin_code(path, num_extras=0, num_var_length_extras=0, extra_bytes=2, var_length_extra_bytes=None):

    with open(path, "rb") as compressed_file:
        compressed = ''.join(["{:08b}".format(x) for x in compressed_file.read()])

    extra_bits = compressed[:num_extras * extra_bytes * 8]
    compressed = compressed[num_extras * extra_bytes * 8:]
    
    extras = [int('0b' + extra_bits[s:s + extra_bytes * 8], 2) for s in range(0, 
                                                                              num_extras * extra_bytes * 8, 
                                                                              extra_bytes * 8)]
    
    var_length_extras = []
    
    for var_extra_byte_size in var_length_extra_bytes:
        
        # Read length of current extra
        extra_length = int('0b' + compressed[:extra_bytes * 8], 2)
        extra = [int('0b' + compressed[extra_bytes * 8 + s:extra_bytes * 8 + s + var_extra_byte_size * 8], 2) 
                 for s in range(0, extra_length * var_extra_byte_size * 8, var_extra_byte_size * 8)]
        
        compressed = compressed[extra_bytes * 8 + extra_length * var_extra_byte_size * 8:]
        
        var_length_extras.append(extra)

    return compressed, extras, var_length_extras