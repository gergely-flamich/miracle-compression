import numpy as np

# ==============================================================================
# Helper Functions
# ==============================================================================

def write_bin_code(code, path, extras=None, var_length_extras=None, var_length_bytes=3):

    # Pad the code
    code += "0" * (8 - len(code) % 8) if len(code) % 8 != 0 else ""

    message_bytes = [int('0b' + code[s:s + 8], 2) for s in range(0, len(code), 8)]

    with open(path, "wb") as compressed_file:
        
        if extras is not None:
            for extra in extras:
                compressed_file.write(bytes([extra // 256, extra % 256]))
                
        if var_length_extras is not None:
            for extra in var_length_extras:
                
                # First code the length of the extra on 16 bits
                compressed_file.write(bytes([len(extra) // 256, len(extra) % 256]))
                
                # Write the rest on var_length_bytes bytes
                for item in extra:
                    compressed_file.write(bytes([item // (256 * 256), item // 256, item % 256]))
                
        
        compressed_file.write(bytes(message_bytes))


def read_bin_code(path, num_extras=0, num_var_length_extras=0):

    with open(path, "rb") as compressed_file:
        compressed = ''.join(["{:08b}".format(x) for x in compressed_file.read()])

    extra_bits = compressed[:num_extras * 16]
    compressed = compressed[num_extras * 16:]
    
    extras = [int('0b' + extra_bits[s:s + 16], 2) for s in range(0, num_extras * 16, 16)]
    
    var_length_extras = []
    
    for i in range(num_var_length_extras):
        
        # Read length of current extra
        extra_length = int('0b' + compressed[:16], 2)
        extra = [int('0b' + extra_bits[16 + s:16 + s], 2) for s in range(0, extra_length * 24, 24)]
        
        compressed = compressed[16 + extra_length * 24:]
        
        var_length_extras.append(extra)
    
    return compressed, extras, var_length_extras