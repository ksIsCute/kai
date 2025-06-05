#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curl/curl.h>

#define MAX_VOCAB_SIZE 10000
#define MAX_SENTENCE_LENGTH 1000
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
        // Clamp to prevent overflow
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
        
        // Find max for numerical stability
        for (int j = 0; j < logits->cols; j++) {
            if (logits->data[i * logits->cols + j] > max_logit) {
                max_logit = logits->data[i * logits->cols + j];
            }
        }
        
        // Compute exponentials and sum
        for (int j = 0; j < logits->cols; j++) {
            float exp_val = exp(logits->data[i * logits->cols + j] - max_logit);
            probs.data[i * probs.cols + j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
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
    
    // Create temporary matrices for gates
    Matrix i_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix f_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix g_gate = create_matrix(1, HIDDEN_SIZE);
    Matrix o_gate = create_matrix(1, HIDDEN_SIZE);
    
    // Split gates
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        i_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j]));
        f_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j + HIDDEN_SIZE]));
        g_gate.data[j] = tanh(xh.data[j + 2 * HIDDEN_SIZE]);
        o_gate.data[j] = 1.0 / (1.0 + exp(-xh.data[j + 3 * HIDDEN_SIZE]));
    }
    
    // Update cell state
    Matrix c_new = create_matrix(1, HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        c_new.data[j] = f_gate.data[j] * c_prev->data[j] + i_gate.data[j] * g_gate.data[j];
    }
    
    // Update hidden state
    Matrix h_new = create_matrix(1, HIDDEN_SIZE);
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        h_new.data[j] = o_gate.data[j] * tanh(c_new.data[j]);
    }
    
    // Copy to previous states
    memcpy(h_prev->data, h_new.data, HIDDEN_SIZE * sizeof(float));
    memcpy(c_prev->data, c_new.data, HIDDEN_SIZE * sizeof(float));
    
    // Free temporary matrices
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
    return probs->cols - 1; // fallback
}

// Curl callback to write data
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
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        // Create some default training data if fetch fails
        printf("Using default training data...\n");
        training_size = 5;
        training_data = malloc(training_size * sizeof(char*));
        training_data[0] = strdup("hello world how are you");
        training_data[1] = strdup("this is a test sentence");
        training_data[2] = strdup("machine learning is interesting");
        training_data[3] = strdup("neural networks are powerful");
        training_data[4] = strdup("artificial intelligence is amazing");
        curl_easy_cleanup(curl);
        return;
    }
    
    if (chunk.size == 0) {
        printf("No data fetched, using default training data...\n");
        training_size = 5;
        training_data = malloc(training_size * sizeof(char*));
        training_data[0] = strdup("hello world how are you");
        training_data[1] = strdup("this is a test sentence");
        training_data[2] = strdup("machine learning is interesting");
        training_data[3] = strdup("neural networks are powerful");
        training_data[4] = strdup("artificial intelligence is amazing");
        curl_easy_cleanup(curl);
        if (chunk.data) free(chunk.data);
        return;
    }
    
    // Count lines
    training_size = 0;
    for (size_t i = 0; i < chunk.size; i++) {
        if (chunk.data[i] == '\n') training_size++;
    }
    if (training_size == 0) training_size = 1; // At least one line
    
    // Allocate and store sentences
    training_data = malloc(training_size * sizeof(char*));
    char* data_copy = strdup(chunk.data);
    char* line = strtok(data_copy, "\n");
    int i = 0;
    while (line && i < training_size) {
        training_data[i] = strdup(line);
        line = strtok(NULL, "\n");
        
        // Update max sequence length
        int len = 0;
        char* temp = strdup(training_data[i]);
        char* token = strtok(temp, " \t\r");
        while (token) {
            len++;
            token = strtok(NULL, " \t\r");
        }
        free(temp);
        if (len > max_sequence_length) max_sequence_length = len;
        i++;
    }
    training_size = i; // Actual number of lines processed
    
    curl_easy_cleanup(curl);
    free(chunk.data);
    free(data_copy);
}

// Process training data
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
}

// Generate new sentences
char* generate_sentence(int max_length, float temperature) {
    if (vocab.size == 0) {
        return strdup("No vocabulary available");
    }
    
    Matrix h = create_matrix(1, HIDDEN_SIZE);
    Matrix c = create_matrix(1, HIDDEN_SIZE);
    
    char* sentence = malloc(MAX_SENTENCE_LENGTH);
    sentence[0] = '\0';
    
    // Start with random word
    int current_word = rand() % vocab.size;
    strncpy(sentence, vocab.entries[current_word].word, MAX_SENTENCE_LENGTH - 1);
    sentence[MAX_SENTENCE_LENGTH - 1] = '\0';
    
    for (int i = 1; i < max_length && strlen(sentence) < MAX_SENTENCE_LENGTH - 50; i++) {
        Matrix x = get_embedding(current_word);
        Matrix h_new = lstm_forward_step(&x, &h, &c);
        
        Matrix logits = matrix_multiply(&h_new, &output_weights);
        matrix_add(&logits, &output_bias);
        
        // Apply temperature
        if (temperature != 1.0 && temperature > 0.0) {
            for (int j = 0; j < logits.cols; j++) {
                logits.data[j] /= temperature;
            }
        }
        
        Matrix probs = softmax(&logits);
        int next_word = sample_from_distribution(&probs);
        
        // Add space and next word
        strncat(sentence, " ", MAX_SENTENCE_LENGTH - strlen(sentence) - 1);
        strncat(sentence, vocab.entries[next_word].word, MAX_SENTENCE_LENGTH - strlen(sentence) - 1);
        
        current_word = next_word;
        
        // Free temporary matrices
        free_matrix(&x);
        free_matrix(&h_new);
        free_matrix(&logits);
        free_matrix(&probs);
        
        // Stop if we hit a period
        if (strcmp(vocab.entries[current_word].word, ".") == 0) break;
    }
    
    free_matrix(&h);
    free_matrix(&c);
    return sentence;
}

// Simple training simulation (without actual backpropagation)
void simulate_training() {
    printf("Simulating training process...\n");
    // In a real implementation, you would implement backpropagation
    // For now, we just simulate the process
    for (int epoch = 0; epoch < 3; epoch++) {
        printf("Epoch %d completed\n", epoch + 1);
    }
}

// Self-training loop (simplified)
void self_train() {
    for (int cycle = 0; cycle < GENERATION_CYCLES; cycle++) {
        printf("\nTraining cycle %d...\n", cycle + 1);
        
        // Generate some new sentences
        printf("Generating new sentences...\n");
        for (int i = 0; i < 3; i++) {
            char* sentence = generate_sentence(8 + rand() % 5, 0.8);
            printf("Generated: %s\n", sentence);
            free(sentence);
        }
        
        // Simulate training
        simulate_training();
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
    
    if (vocab.size > 0) {
        init_network();
        
        // Self-training loop
        printf("Starting self-training...\n");
        self_train();
        
        // Generate final output
        printf("\nFinal generated sentences:\n");
        for (int i = 0; i < 5; i++) {
            char* sentence = generate_sentence(10, 0.7);
            printf("%d. %s\n", i+1, sentence);
            free(sentence);
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
    
    free_matrix(&embedding_weights);
    free_matrix(&lstm_weights_xh);
    free_matrix(&lstm_weights_hh);
    free_matrix(&lstm_bias);
    free_matrix(&output_weights);
    free_matrix(&output_bias);
    
    return 0;
}