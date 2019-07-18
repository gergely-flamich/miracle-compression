
library(binaryLogic)

seed <- 1
set.seed(seed)

# Function that draws greedily a sample with high target value

sample_greedy <- function(m_q, v_q, n_bits, n_steps) {

    m_p <- 0
    v_p <- 1
    group_size <- length(m_q)
    n_samples <- 2^(n_bits / n_steps)

    sample_index <- c()
    best_sample <- rep(0, group_size)
    for (i in 1 : n_steps) {
        set.seed(1e3 * seed + i)
        samples <- matrix(best_sample, n_samples, group_size, byrow = T) + m_p / n_steps + sqrt(v_p / n_steps) * matrix(rnorm(n_samples * group_size), n_samples, group_size)
        a <- apply(samples, 1, function(x) sum(dnorm(x, m_q, sqrt(v_q), log = T)))
        index <- which.max(a)
        best_sample <- samples[ index, ]
        sample_index <- c(sample_index, as.binary(index - 1, n = n_bits / n_steps))
    }

    # we return an integer encoding the generated sample, together with the sample value

    list(sum(2^seq(n_bits - 1, 0) * sample_index), best_sample)
}

# We generate Gaussian encoding and random target disributions of dimension
# 100,000 represented by their means and variances. The means and variances of
# targets are adjusted so that the number of samples used by the sampler are
# less than 256 so that we encode the samples at least in a byte.

latent_dimension <- 100000
m_p <- 0
v_p <- 1
m_q <- 0.1 * rnorm(latent_dimension)
v_q <- exp(0.7 * rnorm(latent_dimension))

# Artificially increase the deviation from the prior for the first two dimensions

m_q[ 1 ] <- 8
v_q[ 1 ] <- 0.1

m_q[ 2 ] <- -8
v_q[ 2 ] <- 0.1

# We compute the KL divergence per dimension

KL <- log(sqrt(v_p) / sqrt(v_q)) + (v_q + (m_q - m_p)^2) / (2 * v_p) - 0.5

# Report an estimate of the theoretical compression bound in bytes

theoretical_size_in_bytes <- ceiling((sum(KL) + 2 * log(sum(KL) + 1)) / log(2)) / 8

cat("Theoretical size in bytes: ", theoretical_size_in_bytes, "\n")

# We iterate finding blocks as large as possible and sampling from them

result <- c()
indexes <- c()
sizes <- c()
n_samples_per_block <- c()
final_sample <- c()

n_bits_index_encoding <- 300
n_bits_size_encoding <- 11
n_steps <- 100
done <- FALSE
i <- 1
while (i < length(m_q)) {

    # We determine the group size, should be as large as possible within the established limits

    group_size <- 1
    n_samples_local <- ceiling(exp(sum(KL[ i : (i + group_size - 1) ]))) - 2
    while ((n_samples_local < 2^n_bits_index_encoding) && (i + group_size - 1 < length(m_q)) && group_size < 2^n_bits_size_encoding) {
        group_size <- group_size + 1
        n_samples_local <- ceiling(exp(sum(KL[ i : (i + group_size - 1) ]))) - 2
    }
    group_size <- group_size - 1

    # We generate the sample greedily

    ret <- sample_greedy(m_q[ i : (i + group_size - 1) ], v_q[ i : (i + group_size - 1) ], n_bits_index_encoding, n_steps)

    final_sample <- c(final_sample, ret[[ 2 ]])
    indexes <- c(indexes, ret[[ 1 ]])
    sizes <- c(sizes, group_size)

    i <- i + group_size

    print(i)
}
result <- c(indexes, sizes)

cat("Compression efficiency:", (length(indexes) * n_bits_index_encoding + length(sizes) * n_bits_size_encoding) / 8 / theoretical_size_in_bytes, "\n")
cat("Avg. log-likelihood per dimension of encoded vector:", mean(dnorm(final_sample, m_q, sqrt(v_q), log = T)), "\n")
cat("Avg. log-likelihood per dimension of random vector:", mean(dnorm(m_q + rnorm(length(m_q)) * sqrt(v_q), m_q, sqrt(v_q), log = T)), "\n")
cat("Avg. log-likelihood per dimension of target mean:", mean(dnorm(m_q, m_q, sqrt(v_q), log = T)), "\n")

