// DecisionTree.hpp
#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <tuple>

/*
    Function: build_decision_tree(D, features)
    input: 
        D - dataset
        features - list of available features
    if all instances in D belong to the same class:
        create a leaf node with the class label
        return
    if features is empty:
        Create a leaf node with the most common class label in D
        return
    best_feature <- select the feature with the highest Information Gain
    create a new node with (best_feature)
    for each value (v ∈ values of best_feature):
        subset <- the subset of D where best_feature = v
        if subset is not empty:
            add a new child node and call Build_Decision_Tree(subset, Features - {best_feature})
        else:
            create a leaf node with the most common class label in D  
    return: node
*/
struct Node {
    bool isLeaf;
    int featureIndex;
    double threshold;
    int label;
    std::shared_ptr<Node> left;
    std::shared_ptr<Node> right;

    Node(bool isLeaf, int label = -1, int featureIndex = -1, double threshold = 0.0)
        : isLeaf(isLeaf), label(label), featureIndex(featureIndex), threshold(threshold) {}
};

class DecisionTree {
public:
    std::shared_ptr<Node> root;
    void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
        root = buildTree(data, labels);
    }
    int predict(const std::vector<double>& sample) {
        return predictRecursive(root, sample);
    }

private:
    std::shared_ptr<Node> buildTree(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) {
        if (isPure(labels)) {
            return std::make_shared<Node>(true, labels[0]);
        }

        int bestFeature = -1;
        double bestThreshold = 0.0;
        double bestInfoGain = 0.0;

        for (int feature = 0; feature < data[0].size(); ++feature) {
            for (const auto& sample : data) {
                double threshold = sample[feature];
                auto [leftLabels, rightLabels] = splitLabels(data, labels, feature, threshold);
                double infoGain = calculateInformationGain(labels, leftLabels, rightLabels);

                if (infoGain > bestInfoGain) {
                    bestInfoGain = infoGain;
                    bestFeature = feature;
                    bestThreshold = threshold;
                }
            }
        }

        if (bestFeature == -1) {
            return std::make_shared<Node>(true, majorityLabel(labels));
        }

        auto [leftData, leftLabels, rightData, rightLabels] = splitData(data, labels, bestFeature, bestThreshold);
        auto leftNode = buildTree(leftData, leftLabels);
        auto rightNode = buildTree(rightData, rightLabels);

        auto node = std::make_shared<Node>(false, -1, bestFeature, bestThreshold);
        node->left = leftNode;
        node->right = rightNode;

        return node;
    }

    int predictRecursive(std::shared_ptr<Node> node, const std::vector<double>& sample) {
        if (node->isLeaf) return node->label;

        if (sample[node->featureIndex] <= node->threshold)
            return predictRecursive(node->left, sample);
        else
            return predictRecursive(node->right, sample);
    }

    double calculateEntropy(const std::vector<int>& labels) {
        std::map<int, int> labelCounts;
        for (int label : labels) labelCounts[label]++;
        /*
        Entropy(S)=−∑i=1|C|pilog2(pi)
        where C is the set of classes in the dataset S, and
        pi is the proportion of data belonging to each class in the set S.  
        */
        double entropy = 0.0;
        for (const auto& [label, count] : labelCounts) {
            double p = static_cast<double>(count) / labels.size();
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double calculateInformationGain(const std::vector<int>& parentLabels, const std::vector<int>& leftLabels, const std::vector<int>& rightLabels) {
        double parentEntropy = calculateEntropy(parentLabels);
        double leftEntropy = calculateEntropy(leftLabels);
        double rightEntropy = calculateEntropy(rightLabels);
        /*
            Information Gain(S, A) = Entropy(S) - ∑v∈Values(A) (|Sv| / |S|) * Entropy(Sv)
            where S is the dataset, A is the feature, and Sv is the subset of S for which feature A has value v.
        */
        double weightLeft = static_cast<double>(leftLabels.size()) / parentLabels.size();
        double weightRight = static_cast<double>(rightLabels.size()) / parentLabels.size();

        return parentEntropy - (weightLeft * leftEntropy + weightRight * rightEntropy);
    }


    std::tuple<std::vector<int>, std::vector<int>> splitLabels(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int featureIndex, double threshold) {
        std::vector<int> leftLabels, rightLabels;
        for (int i = 0; i < data.size(); ++i) {
            if (data[i][featureIndex] <= threshold)
                leftLabels.push_back(labels[i]);
            else
                rightLabels.push_back(labels[i]);
        }
        return {leftLabels, rightLabels};
    }
    

    std::tuple<std::vector<std::vector<double>>, std::vector<int>, std::vector<std::vector<double>>, std::vector<int>>
    splitData(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, int featureIndex, double threshold) {
        std::vector<std::vector<double>> leftData, rightData;
        std::vector<int> leftLabels, rightLabels;
        for (int i = 0; i < data.size(); ++i) {
            if (data[i][featureIndex] <= threshold) {
                leftData.push_back(data[i]);
                leftLabels.push_back(labels[i]);}
                else{
                rightData.push_back(data[i]);
                rightLabels.push_back(labels[i]);}}
        return {leftData, leftLabels, rightData, rightLabels};
    }
    bool isPure(const std::vector<int>& labels) {
        /*
            Gini impurity = 1 - ∑(p_i)^2
            where p_i is the proportion of data labeled with class i.
        */
        for (int i = 1; i < labels.size(); ++i) {   
            if (labels[i] != labels[0]) return false;
        }
        return true;
    }
    int majorityLabel(const std::vector<int>& labels) {
        std::map<int, int> labelCounts;
        for (int label : labels) labelCounts[label]++;
        
        int majorityLabel = labels[0];
        int maxCount = 0;
        for (const auto& [label, count] : labelCounts) {
            if (count > maxCount) {
                maxCount = count;
                majorityLabel = label;
            }
        }
        return majorityLabel;
    }
};

#endif // DECISION_TREE_HPP
