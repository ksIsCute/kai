#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curl/curl.h>

#define MAX_VOCAB_SIZE 10000
#define MAX_SENTENCE_LENGTH 100
#define EMBEDDING_SIZE 64
#define HIDDEN_SIZE 128
#define LEARNING_RATE 0.01
#define EPOCHS 5
#define GENERATION_CYCLES 3
#define BATCH_SIZE 16
#define SEQUENCE_LENGTH 20

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

// Memory buffer for curl
typedef struct {
    char* data;
    size_t size;
} MemoryBuffer;

// Matrix operations
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(rows * cols, sizeof(float));
    return m;
}

void random_init_matrix(Matrix* m, float scale) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = scale * (2.0 * rand() / RAND_MAX - 1.0);
    }
}

void free_matrix(Matrix* m) {
    free(m->data);
}

Matrix matrix_multiply(const Matrix* a, const Matrix* b) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Matrix dimensions mismatch for multiplication\n");
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
        fprintf(stderr, "Matrix dimensions mismatch for addition\n");
        exit(1);
    }
    
    for (int i = 0; i < a->rows * a->cols; i++) {
        a->data[i] += b->data[i];
    }
}

Matrix matrix_add_const(const Matrix* a, float c) {
    Matrix result = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows * a->cols; i++) {
        result.data[i] = a->data[i] + c;
    }
    return result;
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
        result.data[i] = 1.0 / (1.0 + exp(-m->data[i]));
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

Matrix matrix_concat(const Matrix* a, const Matrix* b, int axis) {
    if (axis == 0) { // Vertical concatenation
        if (a->cols != b->cols) {
            fprintf(stderr, "Matrix column mismatch for concatenation\n");
            exit(1);
        }
        Matrix result = create_matrix(a->rows + b->rows, a->cols);
        memcpy(result.data, a->data, a->rows * a->cols * sizeof(float));
        memcpy(result.data + a->rows * a->cols, b->data, b->rows * b->cols * sizeof(float));
        return result;
    } else { // Horizontal concatenation
        if (a->rows != b->rows) {
            fprintf(stderr, "Matrix row mismatch for concatenation\n");
            exit(1);
        }
        Matrix result = create_matrix(a->rows, a->cols + b->cols);
        for (int i = 0; i < a->rows; i++) {
            memcpy(&result.data[i * result.cols], &a->data[i * a->cols], a->cols * sizeof(float));
            memcpy(&result.data[i * result.cols + a->cols], &b->data[i * b->cols], b->cols * sizeof(float));
        }
        return result;
    }
}

// Vocabulary functions
void init_vocabulary() {
    vocab.entries = malloc(MAX_VOCAB_SIZE * sizeof(VocabEntry));
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
    // Initialize embedding layer
    embedding_weights = create_matrix(vocab.size, EMBEDDING_SIZE);
    random_init_matrix(&embedding_weights, 0.1);
    
    // Initialize LSTM weights
    lstm_weights_xh = create_matrix(EMBEDDING_SIZE, 4 * HIDDEN_SIZE);
    lstm_weights_hh = create_matrix(HIDDEN_SIZE, 4 * HIDDEN_SIZE);
    lstm_bias = create_matrix(1, 4 * HIDDEN_SIZE);
    random_init_matrix(&lstm_weights_xh, 0.1);
    random_init_matrix(&lstm_weights_hh, 0.1);
    random_init_matrix(&lstm_bias, 0.1);
    
    // Initialize output layer
    output_weights = create_matrix(HIDDEN_SIZE, vocab.size);
    output_bias = create_matrix(1, vocab.size);
    random_init_matrix(&output_weights, 0.1);
    random_init_matrix(&output_bias, 0.1);
}

Matrix get_embedding(int word_index) {
    Matrix embedding = create_matrix(1, EMBEDDING_SIZE);
    memcpy(embedding.data, &embedding_weights.data[word_index * EMBEDDING_SIZE], 
           EMBEDDING_SIZE * sizeof(float));
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
            probs.data[i * probs.cols + j] = exp(logits->data[i * logits->cols + j] - max_logit);
            sum_exp += probs.data[i * probs.cols + j];
        }
        
        for (int j = 0; j < logits->cols; j++) {
            probs.data[i * probs.cols + j] /= sum_exp;
        }
    }
    return probs;
}

Matrix lstm_forward_step(const Matrix* x, Matrix* h_prev, Matrix* c_prev) {
    // Compute gates
    Matrix xh = matrix_multiply(x, &lstm_weights_xh);
    Matrix hh = matrix_multiply(h_prev, &lstm_weights_hh);
    matrix_add(&xh, &hh);
    matrix_add(&xh, &lstm_bias);
    
    // Split into input, forget, cell, output gates
    Matrix gates = xh;
    Matrix i = matrix_sigmoid(&(Matrix){gates.data, 1, HIDDEN_SIZE});
    Matrix f = matrix_sigmoid(&(Matrix){gates.data + HIDDEN_SIZE, 1, HIDDEN_SIZE});
    Matrix g = matrix_tanh(&(Matrix){gates.data + 2 * HIDDEN_SIZE, 1, HIDDEN_SIZE});
    Matrix o = matrix_sigmoid(&(Matrix){gates.data + 3 * HIDDEN_SIZE, 1, HIDDEN_SIZE});
    
    // Update cell state
    Matrix c = matrix_hadamard(&f, c_prev);
    Matrix temp = matrix_hadamard(&i, &g);
    matrix_add(&c, &temp);
    
    // Update hidden state
    Matrix tanh_c = matrix_tanh(&c);
    Matrix h = matrix_hadamard(&o, &tanh_c);
    
    // Copy to previous states
    memcpy(h_prev->data, h.data, HIDDEN_SIZE * sizeof(float));
    memcpy(c_prev->data, c.data, HIDDEN_SIZE * sizeof(float));
    
    // Free temporary matrices
    free_matrix(&xh);
    free_matrix(&hh);
    free_matrix(&i);
    free_matrix(&f);
    free_matrix(&g);
    free_matrix(&o);
    free_matrix(&temp);
    free_matrix(&tanh_c);
    
    return h;
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
    return probs->cols - 1; // fallback
}

// Curl callback to write data
size_t write_callback(void* contents, size_t size, size_t nmemb, MemoryBuffer* mem) {
    size_t realsize = size * nmemb;
    mem->data = realloc(mem->data, mem->size + realsize + 1);
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;
    return realsize;
}

// Fetch training data from GitHub
void fetch_training_data(const char* url) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        exit(1);
    }

    MemoryBuffer chunk = {0};
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&chunk);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
    
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        exit(1);
    }
    
    // Count lines
    char* ptr = chunk.data;
    while (*ptr) {
        if (*ptr++ == '\n') training_size++;
    }
    
    // Allocate and store sentences
    training_data = malloc(training_size * sizeof(char*));
    ptr = chunk.data;
    char* line = strtok(ptr, "\n");
    for (int i = 0; i < training_size && line; i++) {
        training_data[i] = strdup(line);
        line = strtok(NULL, "\n");
        
        // Update max sequence length
        int len = 0;
        char* temp = strdup(training_data[i]);
        char* token = strtok(temp, " ");
        while (token) {
            len++;
            token = strtok(NULL, " ");
        }
        free(temp);
        if (len > max_sequence_length) max_sequence_length = len;
    }
    
    curl_easy_cleanup(curl);
    free(chunk.data);
}

// Process training data
void process_training_data() {
    for (int i = 0; i < training_size; i++) {
        char* temp = strdup(training_data[i]);
        char* token = strtok(temp, " ");
        while (token) {
            add_word_to_vocab(token);
            token = strtok(NULL, " ");
        }
        free(temp);
    }
}

// Generate new sentences
char* generate_sentence(int max_length, float temperature) {
    Matrix h = create_matrix(1, HIDDEN_SIZE);
    Matrix c = create_matrix(1, HIDDEN_SIZE);
    
    char* sentence = malloc(MAX_SENTENCE_LENGTH);
    sentence[0] = '\0';
    
    // Start with random word
    int current_word = rand() % vocab.size;
    strcpy(sentence, vocab.entries[current_word].word);
    
    for (int i = 1; i < max_length; i++) {
        Matrix x = get_embedding(current_word);
        Matrix h_new = lstm_forward_step(&x, &h, &c);
        
        Matrix logits = matrix_multiply(&h_new, &output_weights);
        matrix_add(&logits, &output_bias);
        
        // Apply temperature
        if (temperature != 1.0) {
            for (int j = 0; j < logits.cols; j++) {
                logits.data[j] /= temperature;
            }
        }
        
        Matrix probs = softmax(&logits);
        int next_word = sample_from_distribution(&probs);
        
        // Add space between words
        strcat(sentence, " ");
        strcat(sentence, vocab.entries[next_word].word);
        
        current_word = next_word;
        
        // Free temporary matrices
        free_matrix(&x);
        free_matrix(&h_new);
        free_matrix(&logits);
        free_matrix(&probs);
        
        // Stop if we hit end token
        if (strcmp(vocab.entries[current_word].word, ".") == 0) break;
    }
    
    free_matrix(&h);
    free_matrix(&c);
    return sentence;
}

// Training functions
float calculate_loss(int* sequence, int seq_length) {
    Matrix h = create_matrix(1, HIDDEN_SIZE);
    Matrix c = create_matrix(1, HIDDEN_SIZE);
    float loss = 0.0;
    
    for (int t = 0; t < seq_length - 1; t++) {
        Matrix x = get_embedding(sequence[t]);
        Matrix h_new = lstm_forward_step(&x, &h, &c);
        
        Matrix logits = matrix_multiply(&h_new, &output_weights);
        matrix_add(&logits, &output_bias);
        
        Matrix probs = softmax(&logits);
        
        // Cross-entropy loss
        loss += -log(probs.data[sequence[t+1]]);
        
        free_matrix(&x);
        free_matrix(&h_new);
        free_matrix(&logits);
        free_matrix(&probs);
    }
    
    free_matrix(&h);
    free_matrix(&c);
    return loss / (seq_length - 1);
}

void train_on_batch(int** batch, int batch_size) {
    // Simplified training - in a real implementation you'd need to:
    // 1. Implement backpropagation through time
    // 2. Update weights properly
    // This is a placeholder for the actual training logic
    
    for (int i = 0; i < batch_size; i++) {
        float loss = calculate_loss(batch[i], SEQUENCE_LENGTH);
        printf("Batch %d loss: %.4f\n", i, loss);
    }
}

// Self-training loop
void self_train() {
    for (int cycle = 0; cycle < GENERATION_CYCLES; cycle++) {
        printf("\nTraining cycle %d...\n", cycle + 1);
        
        // Generate new sentences
        int new_sentences = training_size / 2; // Generate half as many as original
        char** new_data = malloc(new_sentences * sizeof(char*));
        
        for (int i = 0; i < new_sentences; i++) {
            new_data[i] = generate_sentence(10 + rand() % 10, 0.5); // With temperature
            printf("Generated: %s\n", new_data[i]);
        }
        
        // Add to training data
        char** combined = malloc((training_size + new_sentences) * sizeof(char*));
        memcpy(combined, training_data, training_size * sizeof(char*));
        memcpy(combined + training_size, new_data, new_sentences * sizeof(char*));
        
        free(training_data);
        training_data = combined;
        training_size += new_sentences;
        
        // Re-process vocabulary
        vocab.size = 0; // Reset vocabulary
        process_training_data();
        
        // Re-initialize network with new vocab size
        free_matrix(&embedding_weights);
        free_matrix(&lstm_weights_xh);
        free_matrix(&lstm_weights_hh);
        free_matrix(&lstm_bias);
        free_matrix(&output_weights);
        free_matrix(&output_bias);
        init_network();
        
        // Train on the new data (simplified)
        printf("Training on new data...\n");
        // In a real implementation, you would:
        // 1. Convert text to sequences of word indices
        // 2. Create batches
        // 3. Train properly with backpropagation
    }
}

int main() {
    srand(time(NULL));
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Fetch training data from GitHub
    printf("Fetching training data...\n");
    fetch_training_data("https://raw.githubusercontent.com/ksiscute/kai/main/training_sentences.txt");
    
    // Process data and initialize network
    printf("Initializing vocabulary and network...\n");
    init_vocabulary();
    process_training_data();
    init_network();
    
    // Self-training loop
    printf("Starting self-training...\n");
    self_train();
    
    // Generate final output
    printf("\nFinal generated sentences:\n");
    for (int i = 0; i < 5; i++) {
        char* sentence = generate_sentence(15, 0.7); // With temperature
        printf("%d. %s\n", i+1, sentence);
        free(sentence);
    }
    
    // Cleanup
    curl_global_cleanup();
    for (int i = 0; i < training_size; i++) {
        free(training_data[i]);
    }
    free(training_data);
    
    for (int i = 0; i < vocab.size; i++) {
        free(vocab.entries[i].word);
    }
    free(vocab.entries);
    
    free_matrix(&embedding_weights);
    free_matrix(&lstm_weights_xh);
    free_matrix(&lstm_weights_hh);
    free_matrix(&lstm_bias);
    free_matrix(&output_weights);
    free_matrix(&output_bias);
    
    return 0;
}