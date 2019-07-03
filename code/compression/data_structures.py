import numpy as np
from sobol_seq import i4_sobol_generate

# ------------------------------------------------------------------------------
# Efficient Storage for the coded A* sampling
# ------------------------------------------------------------------------------

class TreeLeaf:
    def __init__(self, val, idx):
        # Sample value
        self.val = val

        # Sample index
        self.idx = idx

        # Children
        self.left = None
        self.right = None

    def __str__(self):
        return "{} : {}".format(self.val, self.idx)



class IntervalTree:

    def __init__(self, num_nodes):
        self.root = None
        self.idx = 0

        self.depth = 0

        self.node_list = []

        uniform_approx_samp_block = i4_sobol_generate(1, num_nodes, skip=10)

        for i in range(num_nodes):
            self.add(uniform_approx_samp_block[i, 0])


    def add(self, val):

        leaf = TreeLeaf(val, self.idx)

        if self.root is None:
            self.root = leaf
            self.depth = 1
        else:
            current = self.root
            depth = 1

            while True:
                depth += 1

                if val < current.val:

                    if current.left is None:
                        current.left = leaf
                        break
                    else:
                        current = current.left

                elif val > current.val:

                    if current.right is None:
                        current.right = leaf
                        break
                    else:
                        current = current.right

            if self.depth < depth:
                self.depth = depth

        self.idx += 1
        self.node_list.append(leaf)


    def _to_list(self, node):
        if node is not None:

            l = self._to_list(node.left)

            l += [(node.val, node.idx)]

            l += self._to_list(node.right)

            return l
        else:
            return []

    def to_list(self):
        if self.root is None:
            raise Exception("No entries in tree!")

        return self._to_list(self.root)


    def between(self, low, high, cdf=None, inv_cdf=None):
        """
        This is just the lowest common ancestor algorithm, with the 
        extra condition that nodes cannot be their own ancestors
        """

        if self.root is None:
            raise Exception("No entries in tree!")

        if low >= high:
            raise Exception("Low must be less than high!")

        # Make sure we always have an invertible transformation
        if cdf is None or inv_cdf is None:
            cdf = lambda x: x
            inv_cdf = lambda x: x

        low = cdf(low)
        high = cdf(high)

        current = self.root

        while current is not None:

            # If the ancestors still match, go farther down the tree
            if current.val < low and current.val < high:
                current = current.right

            elif current.val > low and current.val > high:
                current = current.left

            # When the ancestors diverge, check three possible cases
            else:
                if current.val == low:
                    current = current.right

                    while current is not None:
                        if current.val < high:
                            break
                        else:
                            should_break = current.val == high

                            current = current.left

                            if should_break: break

                elif current.val == high:
                    current = current.left

                    while current is not None:
                        if current.val > low:
                            break
                        else:
                            should_break = current.val == low

                            current = current.right

                            if should_break: break


                return None if current is None else (inv_cdf(current.val), current.idx)

        return None


    def _pretty_print(self, node, indent):

        if node is not None:

            self._pretty_print(node.left, indent + 2)

            print("  " * indent + " " + str(node),)

            self._pretty_print(node.right, indent + 2)


    def pretty_print(self):
        self._pretty_print(self.root, 0)


# ==============================================================================
# ==============================================================================
# This one is to find the tightest lower bound for AC
# ==============================================================================
# ==============================================================================

class IntervalTreeV2:

    def __init__(self, nodes):

        self.root = None

        self.depth = 0

        # Permute the nodes randomly
        perm = np.random.choice(len(nodes), size=len(nodes), replace=False)

        for i in range(len(perm)):
            self.add(nodes[perm[i]], perm[i])


    def add(self, val, idx):

        leaf = TreeLeaf(val, idx)

        if self.root is None:
            self.root = leaf
            self.depth = 1
        else:
            current = self.root
            depth = 1

            while True:
                depth += 1

                if val < current.val:

                    if current.left is None:
                        current.left = leaf
                        break
                    else:
                        current = current.left

                elif val > current.val:

                    if current.right is None:
                        current.right = leaf
                        break
                    else:
                        current = current.right

            if self.depth < depth:
                self.depth = depth


    def find_tightest_lower_bound(self, z, transformer=lambda x: x):

        if self.root is None:
            raise Exception("Nothing in the tree!")

        else:
            # Initialise the lower bound
            lower_bound = -np.inf
            lower_bound_index = None

            current = self.root

            while current is not None:

                t = transformer(current.val)

                # Stepping right
                if t <= z:
                    lower_bound = t
                    lower_bound_index = current.idx

                    current = current.right

                # Stepping left
                else:
                    current = current.left

            return lower_bound, lower_bound_index


    def _pretty_print(self, node, indent):

        if node is not None:

            self._pretty_print(node.left, indent + 2)

            print("  " * indent + " " + str(node),)

            self._pretty_print(node.right, indent + 2)


    def pretty_print(self):
        self._pretty_print(self.root, 0)

