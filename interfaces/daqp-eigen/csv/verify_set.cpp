#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <daqp.hpp>

Eigen::MatrixXd filter(Eigen::MatrixXd const& vec, double const threshold = 1e-5);
void saveMatrixCSV(std::string const& filename, Eigen::MatrixXd const& matrix);
Eigen::MatrixXd loadMatrixCSV(std::string const& filename);


#define TEST_TYPE 0
// 0: Full problem, i.e., Coupling + Floating base + 2 SE3 + All joints
// 1: Coupling + Floating base + 2 SE3 + Wheels only
// 2: Coupling + Floating base + All joints

int main() {
    Eigen::MatrixXd matrix = loadMatrixCSV("data/clik_tasks_A.csv");
    Eigen::VectorXd vector = loadMatrixCSV("data/clik_tasks_bu.csv");
    Eigen::VectorXd lower  = loadMatrixCSV("data/clik_tasks_bl.csv");
    Eigen::VectorXi breaks = loadMatrixCSV("data/clik_tasks_break_points.csv").cast<int>();

    assert(vector.isApprox(lower) && "Not only equality constraints!");

    Eigen::VectorXi ind = Eigen::VectorXi::LinSpaced(matrix.rows(), 0, matrix.rows() - 1);
#if TEST_TYPE == 1
    // Coupling + Floating base + 2 SE3 + Wheels only
    ind = (Eigen::VectorXi(matrix.rows() - matrix.cols() + 6 + 8) << ind.head(matrix.rows() - matrix.cols() + 6),
           ind.tail(8))
            .finished();
    matrix                    = matrix(ind, Eigen::all);
    vector                    = vector(ind);
    breaks(breaks.size() - 1) = breaks(breaks.size() - 2) + 8;
#elif TEST_TYPE == 2
    // Coupling + Floating base + All joints
    ind = (Eigen::VectorXi(matrix.rows() - 2 * 6) << ind.head(matrix.rows() - 2 * 6 - matrix.cols() + 6),
           ind.tail(matrix.cols() - 6))
            .finished();
    matrix = matrix(ind, Eigen::all);
    vector = vector(ind);
    breaks = (Eigen::VectorXi(breaks.size() - 2) << breaks.head(2), breaks(4) - 2 * 6).finished();
#endif
    saveMatrixCSV("data/A_reduced.csv", filter(matrix));
    saveMatrixCSV("data/b_reduced.csv", filter(vector));

    auto result = daqp_solve(matrix, vector, vector, (Eigen::VectorXi(breaks.size() + 1) << 0, breaks).finished());
    auto daqp   = result.get_primal();
    std::cout << "DAQP solution: " << filter(daqp).transpose() << std::endl;

    Eigen::VectorXd solution = loadMatrixCSV("data/clik_tasks_solution.csv");
    std::cout << "\nDifference: " << filter(daqp - solution).transpose() << std::endl;
    std::cout << "\nSlacks: " << filter(result.get_slack()).transpose() << std::endl;

    auto active_set = result.get_active_set();
    std::cout << "\nActive set: " << active_set.transpose() << std::endl;
    saveMatrixCSV("data/A_active.csv", filter(matrix(active_set, Eigen::all)));
    saveMatrixCSV("data/b_active.csv", filter(vector(active_set)));

    return daqp.isApprox(solution, 1e-4) ? 0 : 1;
}


Eigen::MatrixXd filter(Eigen::MatrixXd const& vec, double const threshold) {
    return (vec.array().abs() > threshold).select(vec, 0);
}


void saveMatrixCSV(std::string const& filename, Eigen::MatrixXd const& matrix) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
            out << matrix(i, j);
            if (j < matrix.cols() - 1) {
                out << ",";  // Add comma except at the last column
            }
        }
        out << "\n";  // New line at the end of each row
    }
    out.close();
}


Eigen::MatrixXd loadMatrixCSV(std::string const& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Could not open file for reading: " << filename << std::endl;
        return Eigen::MatrixXd();
    }

    std::vector<std::vector<double>> values;
    std::string line, cell;

    while (std::getline(in, line)) {
        std::vector<double> row;
        std::stringstream lineStream(line);
        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        values.push_back(row);
    }

    // Convert vector to Eigen matrix
    if (values.empty()) {
        return Eigen::MatrixXd();
    }

    Eigen::Index rows = values.size();
    Eigen::Index cols = values[0].size();
    Eigen::MatrixXd matrix(rows, cols);
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (Eigen::Index j = 0; j < cols; ++j) {
            matrix(i, j) = values[i][j];
        }
    }

    return matrix;
}
