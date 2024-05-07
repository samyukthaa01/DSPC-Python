#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <numeric>


using namespace std;
using namespace std::chrono;

struct token {
    int offset;
    int length;
    char next;
};

vector<token> lz77_compress(const string& input, int window_size, int buffer_size) {
    vector<token> output;
    int input_length = input.length();
    int i = 0;

    while (i < input_length) {
        int j = max(0, i - window_size);
        int longest_match_length = 0;
        int best_match_offset = 0;

        while (j < i) {
            int k = 0;
            while (i + k < input_length && input[j + k] == input[i + k] && k < buffer_size) {
                k++;
            }
            if (k > longest_match_length) {
                longest_match_length = k;
                best_match_offset = i - j;
            }
            j++;
        }
        if (longest_match_length > 0) {
            output.push_back({ best_match_offset, longest_match_length, input[i + longest_match_length] });
            i += longest_match_length + 1;
        }
        else {
            output.push_back({ 0, 0, input[i] });
            i++;
        }
    }
    return output;
}

void omp_lz77_compress(const string& input, int window_size, int buffer_size, int threads, vector<vector<token>>& array_tokens) {
    omp_set_num_threads(threads);
    int input_length = input.size();
    int num_chunks = min(threads, input_length); // Ensure at least one chunk per thread

    // Calculate chunk size for each thread
    vector<int> chunk_sizes(threads, input_length / threads);
    for (int i = 0; i < input_length % threads; ++i) {
        chunk_sizes[i]++;
    }

    // Allocate space for each thread's token vector
    array_tokens.resize(num_chunks);

#pragma omp parallel
    {
        // Create a thread-local vector to store tokens
        vector<token> local_tokens;

        // Get the thread ID
        int tid = omp_get_thread_num();

        // Calculate start and end index for the current thread
        int start_index = accumulate(chunk_sizes.begin(), chunk_sizes.begin() + tid, 0);
        int end_index = start_index + chunk_sizes[tid];

        // Process the chunk of data assigned to this thread
        local_tokens = lz77_compress(input.substr(start_index, end_index - start_index), window_size, buffer_size);

        // Merge the thread-local tokens into the main vector
#pragma omp critical
        {
            array_tokens[tid] = move(local_tokens);
        }
    }

    // Merge thread-local outputs into a single vector if needed
    if (num_chunks > 1) {
        array_tokens[0] = move(array_tokens[0]); // Move the tokens from the first thread to the main vector
        for (int i = 1; i < num_chunks; ++i) {
            array_tokens[0].insert(array_tokens[0].end(), make_move_iterator(array_tokens[i].begin()), make_move_iterator(array_tokens[i].end()));
        }
        // Resize the main vector to remove empty entries
        array_tokens.resize(1);
    }
}





string omp_lz77_decompress(const vector<vector<token>>& array_tokens) {
    string output;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < array_tokens.size(); ++i) {
        string local_output;
        for (const auto& t : array_tokens[i]) {
            if (t.length == 0) {
                local_output += t.next;
            }
            else {
                int start = local_output.length() - t.offset;
                for (int j = 0; j < t.length; j++) {
                    local_output += local_output[start + j];
                }
                local_output += t.next;
            }
        }
#pragma omp critical
        output += local_output;
    }
    return output;
}

int main() {
    string filename = "C:\\Users\\user\\Documents\\sample-2mb-text-file.txt";
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error: Could not open file '" << filename << "'." << endl;
        return 1;
    }

    string input((istreambuf_iterator<char>(infile)), istreambuf_iterator<char>());
    infile.close();

    if (input.empty()) {
        cerr << "Error: File '" << filename << "' is empty." << endl;
        return 1;
    }

    int NUM_THREADS = 4; // You can adjust the number of threads here
    vector<vector<token>> array_tokens(NUM_THREADS + 1);

    auto start = high_resolution_clock::now();
    omp_lz77_compress(input, 100, 10, NUM_THREADS, array_tokens);
    auto stop = high_resolution_clock::now();

    auto compression_duration = duration_cast<milliseconds>(stop - start);

    int total_tokens = 0;
    for (const auto& tokens : array_tokens) {
        total_tokens += tokens.size();
    }

    cout << "Size of actual file: " << input.size() << " bytes" << endl << endl;
    cout << "Total number of tokens: " << total_tokens << endl << endl;
    cout << "Compression took: " << compression_duration.count() << " milliseconds" << endl << endl;

    // Display a preview of the compressed data
    cout << "Compressed data preview (first 10 tokens):" << endl << endl;
    int count = min(10, total_tokens);
    for (int i = 0; i < count; i++) {
        cout << "Token " << i << ": { Offset: " << array_tokens[0][i].offset
            << ", Length: " << array_tokens[0][i].length
            << ", Next: '" << array_tokens[0][i].next << "' }" << endl;
    }

    // Decompression process
    start = high_resolution_clock::now();
    string decompressed = omp_lz77_decompress(array_tokens);
    stop = high_resolution_clock::now();

    auto decompression_duration = duration_cast<milliseconds>(stop - start);
    cout << "Decompression took: " << decompression_duration.count() << " milliseconds" << endl << endl;
    // Write the decompressed output to a file
    ofstream outfile("decompressed_out.txt");
    if (!outfile) {
        cerr << "Error: Could not open file 'decompressed_out.txt' for writing." << endl;
        return 1;
    }
    outfile << decompressed;
    outfile.close();

    cout << "Decompressed output written to 'decompressed_out.txt'" << endl;

    cout << "Original Input (preview): " << input << endl << endl;
    cout << "Decompressed Output (preview): " << decompressed << endl << endl;

    return 0;
}