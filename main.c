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

// Memory buffer for curl
typedef struct {
    char* data;
    size_t size;
} MemoryBuffer;

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

// Matrix functions
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

Matrix softmax(Matrix* logits) {
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
    }
    
    curl_easy_cleanup(curl);
    free(chunk.data);
}

// Process training data
void process_training_data() {
    for (int i = 0; i < training_size; i++) {
        char* token = strtok(training_data[i], " ");
        while (token) {
            add_word_to_vocab(token);
            token = strtok(NULL, " ");
        }
    }
}

// Generate new sentences
char* generate_sentence(int max_length) {
    Matrix h = create_matrix(1, HIDDEN_SIZE);
    Matrix c = create_matrix(1, HIDDEN_SIZE);
    
    char* sentence = malloc(MAX_SENTENCE_LENGTH);
    sentence[0] = '\0';
    
    // Start with random word
    int current_word = rand() % vocab.size;
    strcpy(sentence, vocab.entries[current_word].word);
    
    for (int i = 1; i < max_length; i++) {
        Matrix input = create_matrix(1, 1);
        input.data[0] = current_word;
        
        Matrix probs = softmax(&input);
        
        float r = (float)rand() / RAND_MAX;
        float cum_prob = 0.0;
        int next_word = 0;
        for (; next_word < vocab.size; next_word++) {
            cum_prob += probs.data[next_word];
            if (r <= cum_prob) break;
        }
        
        // Add space between words
        strcat(sentence, " ");
        strcat(sentence, vocab.entries[next_word].word);
        
        current_word = next_word;
        free(input.data);
        free(probs.data);
        
        // Stop if we hit end token
        if (strcmp(vocab.entries[current_word].word, ".") == 0) break;
    }
    
    free(h.data);
    free(c.data);
    return sentence;
}

// Self-training loop
void self_train() {
    for (int cycle = 0; cycle < GENERATION_CYCLES; cycle++) {
        printf("Training cycle %d...\n", cycle + 1);
        
        // Generate new sentences
        int new_sentences = training_size / 2; // Generate half as many as original
        char** new_data = malloc(new_sentences * sizeof(char*));
        
        for (int i = 0; i < new_sentences; i++) {
            new_data[i] = generate_sentence(10 + rand() % 10); // 10-20 word sentences
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
        free(embedding_weights.data);
        free(lstm_weights_xh.data);
        free(lstm_weights_hh.data);
        free(lstm_bias.data);
        free(output_weights.data);
        free(output_bias.data);
        init_network();
    }
}

int main() {
    srand(time(NULL));
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Fetch training data from GitHub
    fetch_training_data("https://raw.githubusercontent.com/ksiscute/kai/main/training_sentences.txt");
    
    // Process data and initialize network
    init_vocabulary();
    process_training_data();
    init_network();
    
    // Self-training loop
    self_train();
    
    // Generate final output
    printf("\nFinal generated sentences:\n");
    for (int i = 0; i < 5; i++) {
        char* sentence = generate_sentence(15);
        printf("%d. %s\n", i+1, sentence);
        free(sentence);
    }
    
    // Cleanup
    curl_global_cleanup();
    return 0;
}
