#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curl/curl.h>
#include <ctype.h>

#define MAX_VOCAB_SIZE 10000
#define MAX_SENTENCE_LENGTH 1000
#define EMBEDDING_SIZE 640
#define HIDDEN_SIZE 1280
#define LEARNING_RATE 0.3
#define EPOCHS 150
#define GENERATION_CYCLES 5000
#define BATCH_SIZE 32
#define SEQUENCE_LENGTH 250
#define MIN_SIMILARITY_THRESHOLD 0.3
#define RECURSIVE_DEPTH 100

typedef struct {
    char* word;
    int index;
    int count;
} VocabEntry;

typedef struct {
    VocabEntry* entries;
    int size;
    int capacity;
} Vocabulary;

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    char* sentence;
    float fitness_score;
    int word_count;
} GeneratedSentence;

// Global variables
Vocabulary vocab;
Matrix embedding_weights;
Matrix lstm_weights_xh;
Matrix lstm_weights_hh;
Matrix lstm_bias;
Matrix output_weights;
Matrix output_bias;
char** training_data;
int training_size = 0;
int max_sequence_length = 0;
float* word_frequencies;
GeneratedSentence* generation_pool;
int pool_size = 0;

// Memory buffer for curl
typedef struct {
    char* data;
    size_t size;
} MemoryBuffer;

// Matrix operations (keeping existing functions)
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(rows * cols, sizeof(float));
    if (!m.data) {
        fprintf(stderr, "Failed to allocate matrix memory\n");
        exit(1);
    }
    return m;
}

void random_init_matrix(Matrix* m, float scale) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = scale * (2.0 * rand() / RAND_MAX - 1.0);
    }
}

void free_matrix(Matrix* m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
}

Matrix matrix_multiply(const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Matrix dimensions mismatch for multiplication: %dx%d * %dx%d\n", 
                a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }
    
    Matrix result = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

void matrix_add(Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Matrix dimensions mismatch for addition: %dx%d + %dx%d\n", 
                a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }
    
    for (int i = 0; i < a->rows * a->cols; i++) {
        a->data[i] += b->data[i];
    }
}

Matrix matrix_hadamard(const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Matrix dimensions mismatch for Hadamard product\n");
        exit(1);
    }
    
    Matrix result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result.data[i] = a->data[i] * b->data[i];
    }
    return result;
}

Matrix matrix_sigmoid(const Matrix* m) {
    Matrix result = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        float x = m->data[i];
        if (x > 500) x = 500;
        if (x < -500) x = -500;
        result.data[i] = 1.0 / (1.0 + exp(-x));
    }
    return result;
}

Matrix matrix_tanh(const Matrix* m) {
    Matrix result = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows * m->cols; i++) {
        result.data[i] = tanh(m->data[i]);
    }
    return result;
}

// Vocabulary functions
void init_vocabulary() {
    vocab.entries = malloc(MAX_VOCAB_SIZE * sizeof(VocabEntry));
    if (!vocab.entries) {
        fprintf(stderr, "Failed to allocate vocabulary memory\n");
        exit(1);
    }
    vocab.size = 0;
    vocab.capacity = MAX_VOCAB_SIZE;
}

int get_word_index(const char* word) {
    for (int i = 0; i < vocab.size; i++) {
        if (strcmp(vocab.entries[i].word, word) == 0) {
            return i;
        }
    }
    return -1;
}

void add_word_to_vocab(const char* word) {
    if (vocab.size >= vocab.capacity) return;
    
    int index = get_word_index(word);
    if (index != -1) {
        vocab.entries[index].count++;
        return;
    }
    
    vocab.entries[vocab.size].word = strdup(word);
    vocab.entries[vocab.size].index = vocab.size;
    vocab.entries[vocab.size].count = 1;
    vocab.size++;
}

// Neural network functions
void init_network() {
    if (vocab.size == 0) {
        fprintf(stderr, "Cannot initialize network with empty vocabulary\n");
        return;
    }
    
    embedding_weights = create_matrix(vocab.size, EMBEDDING_SIZE);
    random_init_matrix(&embedding_weights, 0.1);
    
    lstm_weights_xh = create_matrix(EMBEDDING_SIZE, 4 * HIDDEN_SIZE);
    lstm_weights_hh = create_matrix(HIDDEN_SIZE, 4 * HIDDEN_SIZE);
    lstm_bias = create_matrix(1, 4 * HIDDEN_SIZE);
    random_init_matrix(&lstm_weights_xh, 0.1);
    random_init_matrix(&lstm_weights_hh, 0.1);
    random_init_matrix(&lstm_bias, 0.1);
    
    output_weights = create_matrix(HIDDEN_SIZE, vocab.size);
    output_bias = create_matrix(1, vocab.size);
    random_init_matrix(&output_weights, 0.1);
    random_init_matrix(&output_bias, 0.1);
}

Matrix get_embedding(int word_index) {
    if (word_index < 0 || word_index >= vocab.size) {
        fprintf(stderr, "Invalid word index: %d\n", word_index);
        exit(1);
    }
    
    Matrix embedding = create_matrix(1, EMBEDDING_SIZE);
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        embedding.data[i] = embedding_weights.data[word_index * EMBEDDING_SIZE + i];
    }
    return embedding;
}

Matrix softmax(const Matrix* logits) {
    Matrix probs = create_matrix(logits->rows, logits->cols);
    for (int i = 0; i < logits->rows; i++) {
        float max_logit = -INFINITY;
        float sum_exp = 0.0;
        
        for (int j = 0; j < logits->cols; j++) {
            if (logits->data[i * logits->cols + j] > max_logit) {
                max_logit = logits->data[i * logits->cols + j];
            }
        }
        
        for (int j = 0; j < logits->cols; j++) {
            float exp_val = exp(logits->data[i * logits->cols + j] - max_logit);
            probs.data[i * probs.cols + j] = exp_val;
            sum_exp += exp_val;
        }
        
        for (int j = 0; j < logits->cols; j++) {
            probs.data[i * probs.cols + j] /= sum_exp;
        }
    }
    return probs;
}

Matrix lstm_forward_step(const Matrix* x, Matrix* h_prev, Matrix* c_prev) {
    Matrix xh = matrix_multiply(x, &lstm_weights_xh);
    Matrix hh = matrix_multiply(h_prev, &lstm_weights_hh);
    matrix_add(&xh, &hh);
    matrix_add(&xh, &lstm_bias);
    
    Matrix i_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix f_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix g_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix o_gate = create_matrix(1, HIDDEN_SIZE);
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        i_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j]));
        f_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j + HIDDEN_SIZE]));
        g_gate.data[j] = tanh(xh.data[j + 2 * HIDDEN_SIZE]);
        o_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j + 3 * HIDDEN_SIZE]));
    }
    
    Matrix c_new = create_matrix(1, HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        c_new.data[j] = f_gate.data[j] * c_prev->data[j] + i_gate.data[j] * g_gate.data[j];
    }
    
    Matrix h_new = create_matrix(1, HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        h_new.data[j] = o_gate.data[j] * tanh(c_new.data[j]);
    }
    
    memcpy(h_prev->data, h_new.data, HIDDEN_SIZE * sizeof(float));
    memcpy(c_prev->data, c_new.data, HIDDEN_SIZE * sizeof(float));
    
    free_matrix(&xh);
    free_matrix(&hh);
    free_matrix(&i_gate);
    free_matrix(&f_gate);
    free_matrix(&g_gate);
    free_matrix(&o_gate);
    free_matrix(&c_new);
    
    return h_new;
}

int sample_from_distribution(const Matrix* probs) {
    float r = (float)rand() / RAND_MAX;
    float cum_prob = 0.0;
    for (int i = 0; i < probs->cols; i++) {
        cum_prob += probs->data[i];
        if (r <= cum_prob) {
            return i;
        }
    }
    return probs->cols - 1;
}

// NEW: Sentence similarity calculation
float calculate_sentence_similarity(const char* sent1, const char* sent2) {
    char* temp1 = strdup(sent1);
    char* temp2 = strdup(sent2);
    
    // Tokenize sentences
    char* words1[100], * words2[100];
    int count1 = 0, count2 = 0;
    
    char* token = strtok(temp1, " \t\r\n");
    while (token && count1 < 100) {
        words1[count1++] = strdup(token);
        token = strtok(NULL, " \t\r\n");
    }
    
    token = strtok(temp2, " \t\r\n");
    while (token && count2 < 100) {
        words2[count2++] = strdup(token);
        token = strtok(NULL, " \t\r\n");
    }
    
    // Calculate Jaccard similarity
    int common_words = 0;
    for (int i = 0; i < count1; i++) {
        for (int j = 0; j < count2; j++) {
            if (strcmp(words1[i], words2[j]) == 0) {
                common_words++;
                break;
            }
        }
    }
    
    float similarity = (float)common_words / (count1 + count2 - common_words);
    
    // Cleanup
    for (int i = 0; i < count1; i++) free(words1[i]);
    for (int i = 0; i < count2; i++) free(words2[i]);
    free(temp1);
    free(temp2);
    
    return similarity;
}

// NEW: Find best matching training sentence
float find_best_match_score(const char* generated_sentence) {
    float best_score = 0.0;
    for (int i = 0; i < training_size; i++) {
        float score = calculate_sentence_similarity(generated_sentence, training_data[i]);
        if (score > best_score) {
            best_score = score;
        }
    }
    return best_score;
}

// NEW: Calculate word frequency distribution fitness
float calculate_word_frequency_fitness(const char* sentence) {
    char* temp = strdup(sentence);
    float fitness = 0.0;
    int word_count = 0;
    
    char* token = strtok(temp, " \t\r\n");
    while (token) {
        int word_idx = get_word_index(token);
        if (word_idx != -1) {
            // Reward using words with appropriate frequency
            float expected_freq = (float)vocab.entries[word_idx].count / training_size;
            fitness += log(expected_freq + 1e-6); // Avoid log(0)
            word_count++;
        }
        token = strtok(NULL, " \t\r\n");
    }
    
    free(temp);
    return word_count > 0 ? fitness / word_count : 0.0;
}

// NEW: Update network weights based on fitness
void update_weights_from_fitness(const char* sentence, float fitness_score) {
    if (fitness_score < MIN_SIMILARITY_THRESHOLD) return;
    
    // Simple weight adjustment based on fitness
    float adjustment = LEARNING_RATE * (fitness_score - 0.5);
    
    // Adjust embedding weights for words in high-fitness sentences
    char* temp = strdup(sentence);
    char* token = strtok(temp, " \t\r\n");
    
    while (token) {
        int word_idx = get_word_index(token);
        if (word_idx != -1) {
            for (int i = 0; i < EMBEDDING_SIZE; i++) {
                embedding_weights.data[word_idx * EMBEDDING_SIZE + i] += adjustment * 0.1;
            }
        }
        token = strtok(NULL, " \t\r\n");
    }
    
    free(temp);
}

// NEW: Generate sentence with fitness evaluation
GeneratedSentence generate_evaluated_sentence(int max_length, float temperature) {
    GeneratedSentence result;
    result.sentence = NULL;
    result.fitness_score = 0.0;
    result.word_count = 0;
    
    if (vocab.size == 0) {
        result.sentence = strdup("No vocabulary available");
        return result;
    }
    
    Matrix h = create_matrix(1, HIDDEN_SIZE);
    Matrix c = create_matrix(1, HIDDEN_SIZE);
    
    char* sentence = malloc(MAX_SENTENCE_LENGTH);
    sentence[0] = '\0';
    
    int current_word = rand() % vocab.size;
    strncpy(sentence, vocab.entries[current_word].word, MAX_SENTENCE_LENGTH - 1);
    sentence[MAX_SENTENCE_LENGTH - 1] = '\0';
    
    for (int i = 1; i < max_length && strlen(sentence) < MAX_SENTENCE_LENGTH - 50; i++) {
        Matrix x = get_embedding(current_word);
        Matrix h_new = lstm_forward_step(&x, &h, &c);
        
        Matrix logits = matrix_multiply(&h_new, &output_weights);
        matrix_add(&logits, &output_bias);
        
        if (temperature != 1.0 && temperature > 0.0) {
            for (int j = 0; j < logits.cols; j++) {
                logits.data[j] /= temperature;
            }
        }
        
        Matrix probs = softmax(&logits);
        int next_word = sample_from_distribution(&probs);
        
        strncat(sentence, " ", MAX_SENTENCE_LENGTH - strlen(sentence) - 1);
        strncat(sentence, vocab.entries[next_word].word, MAX_SENTENCE_LENGTH - strlen(sentence) - 1);
        
        current_word = next_word;
        
        free_matrix(&x);
        free_matrix(&h_new);
        free_matrix(&logits);
        free_matrix(&probs);
        
        if (strcmp(vocab.entries[current_word].word, ".") == 0) break;
    }
    
    // Calculate fitness score
    float similarity_score = find_best_match_score(sentence);
    float frequency_score = calculate_word_frequency_fitness(sentence);
    
    result.sentence = sentence;
    result.fitness_score = (similarity_score + frequency_score) / 2.0;
    
    // Count words
    char* temp = strdup(sentence);
    char* token = strtok(temp, " \t\r\n");
    while (token) {
        result.word_count++;
        token = strtok(NULL, " \t\r\n");
    }
    free(temp);
    
    free_matrix(&h);
    free_matrix(&c);
    
    return result;
}

// NEW: Recursive training function
void recursive_training_loop(int depth, int max_depth) {
    if (depth >= max_depth) {
        printf("Reached maximum recursion depth %d\n", depth);
        return;
    }
    
    printf("\n=== Recursive Training Depth %d ===\n", depth);
    
    // Generate batch of sentences
    GeneratedSentence* batch = malloc(BATCH_SIZE * sizeof(GeneratedSentence));
    float total_fitness = 0.0;
    float best_fitness = 0.0;
    int best_idx = 0;
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        batch[i] = generate_evaluated_sentence(8 + rand() % 7, 0.8 - depth * 0.1);
        total_fitness += batch[i].fitness_score;
        
        if (batch[i].fitness_score > best_fitness) {
            best_fitness = batch[i].fitness_score;
            best_idx = i;
        }
        
        printf("Gen %d (fit: %.3f): %s\n", i+1, batch[i].fitness_score, batch[i].sentence);
    }
    
    float avg_fitness = total_fitness / BATCH_SIZE;
    printf("Average fitness: %.3f, Best fitness: %.3f\n", avg_fitness, best_fitness);
    
    // Update weights based on high-fitness sentences
    int updates = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        if (batch[i].fitness_score > avg_fitness) {
            update_weights_from_fitness(batch[i].sentence, batch[i].fitness_score);
            updates++;
        }
    }
    printf("Updated weights for %d high-fitness sentences\n", updates);
    
    // Store best sentence from this generation
    if (best_fitness > MIN_SIMILARITY_THRESHOLD) {
        printf("★ Best sentence: %s (fitness: %.3f)\n", batch[best_idx].sentence, best_fitness);
        
        // Add best sentence to training pool for next iteration
        if (pool_size < 100) { // Limit pool size
            generation_pool[pool_size].sentence = strdup(batch[best_idx].sentence);
            generation_pool[pool_size].fitness_score = best_fitness;
            generation_pool[pool_size].word_count = batch[best_idx].word_count;
            pool_size++;
        }
    }
    
    // Cleanup current batch
    for (int i = 0; i < BATCH_SIZE; i++) {
        free(batch[i].sentence);
    }
    free(batch);
    
    // Recursive call with improvement condition
    if (best_fitness > MIN_SIMILARITY_THRESHOLD || depth < max_depth) {
        recursive_training_loop(depth + 1, max_depth);
    } else {
        printf("Stopping recursion: fitness threshold not met at depth %d\n", depth);
    }
}

// Curl callback
size_t write_callback(void* contents, size_t size, size_t nmemb, MemoryBuffer* mem) {
    size_t realsize = size * nmemb;
    char* ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) {
        fprintf(stderr, "Not enough memory (realloc returned NULL)\n");
        return 0;
    }
    
    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    return realsize;
}

// Fetch training data
void fetch_training_data(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        printf("Using default training data...\n");
        training_size = 8;
        training_data = malloc(training_size * sizeof(char*));
        training_data[0] = strdup("hello world how are you today");
        training_data[1] = strdup("this is a test sentence for training");
        training_data[2] = strdup("machine learning is very interesting and powerful");
        training_data[3] = strdup("neural networks can learn complex patterns");
        training_data[4] = strdup("artificial intelligence will change the world");
        training_data[5] = strdup("natural language processing helps computers understand text");
        training_data[6] = strdup("deep learning models require lots of training data");
        training_data[7] = strdup("the future of technology looks very promising");
        return;
    }

    // First pass: count lines
    training_size = 0;
    char buffer[MAX_SENTENCE_LENGTH];
    while (fgets(buffer, sizeof(buffer), file)) {
        training_size++;
    }
    rewind(file);

    // Allocate memory
    training_data = malloc(training_size * sizeof(char*));
    if (!training_data) {
        fprintf(stderr, "Failed to allocate training data array\n");
        fclose(file);
        return;
    }

    // Second pass: read lines
    int i = 0;
    max_sequence_length = 0;
    while (fgets(buffer, sizeof(buffer), file) && i < training_size) {
        // Remove newline character if present
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
        }

        // Skip empty lines
        if (strlen(buffer) > 0) {
            training_data[i] = strdup(buffer);
            if (!training_data[i]) {
                fprintf(stderr, "Failed to duplicate line %d\n", i);
                continue;
            }

            // Calculate max sequence length
            char* temp = strdup(training_data[i]);
            char* token = strtok(temp, " \t\r");
            int len = 0;
            while (token) {
                len++;
                token = strtok(NULL, " \t\r");
            }
            free(temp);

            if (len > max_sequence_length) max_sequence_length = len;
            i++;
        }
    }
    training_size = i; // Actual number of loaded sentences
    fclose(file);

    printf("Successfully loaded %d training sentences from %s\n", training_size, filename);
}

void process_training_data() {
    for (int i = 0; i < training_size; i++) {
        char* temp = strdup(training_data[i]);
        char* token = strtok(temp, " \t\r\n");
        while (token) {
            add_word_to_vocab(token);
            token = strtok(NULL, " \t\r\n");
        }
        free(temp);
    }
    printf("Vocabulary size: %d\n", vocab.size);
    printf("Training sentences:\n");
    for (int i = 0; i < training_size; i++) {
        printf("- %s\n", training_data[i]);
    }
}

int main() {
    srand(time(NULL));
    
    // Initialize generation pool
    generation_pool = malloc(100 * sizeof(GeneratedSentence));
    pool_size = 0;
    
    printf("Loading training data...\n");
    fetch_training_data("training_sentences.txt");  // Changed from URL to local filename
    
    printf("Initializing vocabulary and network...\n");
    init_vocabulary();
    process_training_data();
    
    if (vocab.size > 0) {
        init_network();
        
        printf("\n=== Starting Recursive Training Loop ===\n");
        recursive_training_loop(0, RECURSIVE_DEPTH);
        
        printf("\n=== Final Results ===\n");
        printf("Generated sentence pool (%d sentences):\n", pool_size);
        for (int i = 0; i < pool_size; i++) {
            printf("%d. [%.3f] %s\n", i+1, generation_pool[i].fitness_score, generation_pool[i].sentence);
        }
        
        printf("\nFinal high-quality generated sentences:\n");
        for (int i = 0; i < 5; i++) {
            GeneratedSentence final = generate_evaluated_sentence(10, 0.6);
            printf("%d. [fitness: %.3f] %s\n", i+1, final.fitness_score, final.sentence);
            free(final.sentence);
        }
    } else {
        printf("No vocabulary loaded, cannot proceed with training.\n");
    }
    
    // Cleanup
    curl_global_cleanup();
    
    if (training_data) {
        for (int i = 0; i < training_size; i++) {
            if (training_data[i]) free(training_data[i]);
        }
        free(training_data);
    }
    
    if (vocab.entries) {
        for (int i = 0; i < vocab.size; i++) {
            if (vocab.entries[i].word) free(vocab.entries[i].word);
        }
        free(vocab.entries);
    }
    
    if (generation_pool) {
        for (int i = 0; i < pool_size; i++) {
            free(generation_pool[i].sentence);
        }
        free(generation_pool);
    }
    
    free_matrix(&embedding_weights);
    free_matrix(&lstm_weights_xh);
    free_matrix(&lstm_weights_hh);
    free_matrix(&lstm_bias);
    free_matrix(&output_weights);
    free_matrix(&output_bias);
    
    return 0;
}