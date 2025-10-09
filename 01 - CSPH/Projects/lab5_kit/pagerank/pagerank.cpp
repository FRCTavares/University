#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <getopt.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "common/graph.h"
#include "common/grade.h"

#define USE_BINARY_GRAPH 1

constexpr float PageRankDampening = 0.3;
constexpr double PageRankConvergence = 1e-7;

void pageRank(Graph graph, double *solution, double damping, double convergence)
{
    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs
    const int n_nodes = num_nodes(graph);
    double *old_score = new double[n_nodes];
    double *new_score = new double[n_nodes];

    const double equal_prob = 1.0 / n_nodes;
    for (int i = 0; i < n_nodes; ++i)
    {
        old_score[i] = equal_prob;
    }

    const double damping_const = (1.0 - damping) / static_cast<double>(n_nodes);

    bool converged = false;
    while (!converged)
    {
        // Reset the global diff
        double global_diff = 0.0;

        // First calculate the sum of scores from nodes with no outbound links
        double no_outbound_sum = 0.0;
        for (int i = 0; i < n_nodes; ++i)
            if (outgoing_size(graph, i) == 0)
                no_outbound_sum += old_score[i];

        no_outbound_sum /= static_cast<double>(n_nodes);

        // Compute the new score for all nodes
        for (int i = 0; i < n_nodes; ++i)
        {
            // Transfer the score from incoming nodes to this one
            new_score[i] = 0.0f;
            const Vertex *start = incoming_begin(graph, i);
            const Vertex *end = incoming_end(graph, i);
            for (const Vertex *v = start; v != end; ++v)
                new_score[i] += old_score[*v] / static_cast<double>(outgoing_size(graph, *v));

            // Add the scores from nodes with no outbound links
            new_score[i] += no_outbound_sum;
            // Apply damping
            new_score[i] = damping_const + damping * new_score[i];
            // Accumulate the difference from the last iteration
            global_diff += std::abs(new_score[i] - old_score[i]);
            // Write the result
            solution[i] = new_score[i];
        }

        converged = global_diff < convergence;

        // Swap the new and old scores (the pointers)
        std::swap(old_score, new_score);
    }

    delete old_score;
    delete new_score;
}


int main(int argc, char** argv) {

    std::string graph_filename;

    if (argc < 2)
    {
        std::cerr << "Usage: <path/to/graph/file> [num_threads]\n";
        std::cerr << "  To run results for all thread counts: <path/to/graph/file>\n";
        std::cerr << "  Run with a certain number of threads: <path/to/graph/file> <num_threads>\n";
        exit(1);
    }

    int thread_count = -1;
    if (argc == 3)
    {
        thread_count = atoi(argv[2]);
    }

    graph_filename = argv[1];

    Graph graph;

    printf("----------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    if (thread_count > 0)
    {
        thread_count = std::min(thread_count, omp_get_max_threads());
        printf("Running with %d threads\n", thread_count);
    }
    printf("----------------------------------------------------------\n");

    printf("Loading graph...\n");
    if (USE_BINARY_GRAPH) {
        graph = load_graph_binary(graph_filename.c_str());
    } else {
        graph = load_graph(argv[1]);
        printf("storing binary form of graph!\n");
        store_graph_binary(graph_filename.append(".bin").c_str(), graph);
        delete graph;
        exit(1);
    }
    printf("\n");
    printf("Graph stats:\n");
    printf("  Edges: %d\n", graph->num_edges);
    printf("  Nodes: %d\n", graph->num_nodes);

    //Solution sphere
    double* sol_gold;
    sol_gold = (double*)malloc(sizeof(double) * graph->num_nodes);
    std::string gdir = graph_filename.substr(0, graph_filename.find_last_of("/"));
    std::string gfile = graph_filename.substr(graph_filename.find_last_of("/"), graph_filename.length());
    std::string sol_file = gdir.append("/solutions").append(gfile).append(".prsol").c_str();
    
    load_solution_binary(sol_file.c_str(), sol_gold);

    // If we want to run on all threads
    if (thread_count <= -1)
    {
        std::vector<int> num_threads;

        int max_threads = omp_get_max_threads();
        for (int i = 1; i < max_threads; i *= 2) {
          num_threads.push_back(i);
        }
        num_threads.push_back(max_threads);

        double* sol = (double*)malloc(sizeof(double) * graph->num_nodes);
        bool correct_results = true;

        double runtime_baseline = 0.0;

        std::stringstream timing;
        timing << "Threads  Time (Speedup)\n";

        //Loop through num_threads values;
        for (auto n_th: num_threads)
        {
            printf("----------------------------------------------------------\n");
            std::cout << "Running with " << n_th << " threads" << std::endl;
            //Set thread count
            omp_set_num_threads(n_th);

            //Run implementations
            double runtime = omp_get_wtime();
            pageRank(graph, sol, PageRankDampening, PageRankConvergence);
            runtime = omp_get_wtime() - runtime;

           // record single thread times in order to report speedup
            if (n_th == 1)
                runtime_baseline = runtime;

            std::cout << "Testing Correctness of Page Rank\n";
            if (!compareApprox(graph, sol_gold, sol))
                correct_results = false;

            char buf[1024];
            sprintf(buf, "%4d:   %.4f (%.4fx)\n", n_th, runtime, runtime_baseline/runtime);

            timing << buf;
        }

        printf("----------------------------------------------------------\n");
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
        std::cout << "Correctness: " << std::endl;
        if (!correct_results)
            std::cout << "Page Rank is not Correct" << std::endl;
    }
    //Run the code with only one thread count and only report speedup
    else
    {
        double* sol = (double*)malloc(sizeof(double) * graph->num_nodes);
        bool correct_results = true;

        std::stringstream timing;
        timing << "Threads  Time\n";

        // Loop through assignment values;
        std::cout << "Running with " << thread_count << " threads" << std::endl;
        // Set thread count
        omp_set_num_threads(thread_count);

        // Run implementations
        double runtime = omp_get_wtime();
        pageRank(graph, sol, PageRankDampening, PageRankConvergence);
        runtime = omp_get_wtime() - runtime;

        std::cout << "Testing Correctness of Page Rank\n";
        if (!compareApprox(graph, sol_gold, sol)) {
          correct_results = false;
        }

        char buf[1024];
        sprintf(buf, "%4d:   %.4f\n", thread_count, runtime);

        timing << buf;
        if (!correct_results)
            std::cout << "Page Rank is not Correct" << std::endl;
        printf("----------------------------------------------------------\n");
        std::cout << "Your Code: Timing Summary" << std::endl;
        std::cout << timing.str();
        printf("----------------------------------------------------------\n");
    }

    delete graph;

    return 0;
}
