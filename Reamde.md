### Build_Decision_Tree Function

This function recursively builds a decision tree based on a dataset and a list of features.

```python
def Build_Decision_Tree(D, Features):
    """
    Function to build a decision tree.

    Parameters:
        D: dataset containing instances and their corresponding class labels.
        Features: list of features available for splitting the dataset.
    """

    # Check if all instances in D belong to the same class
    if all_instances_same_class(D):
        # Create a leaf node with the class label
        create_leaf_node(class_label=D[0].class_label)
        return

    # If there are no more features to split on
    if not Features:
        # Create a leaf node with the most common class label in D
        create_leaf_node(class_label=most_common_class_label(D))
        return

    # Select the feature with the highest information gain
    best_feature = select_best_feature(Features, D)

    # Create a new node for the decision tree with the selected best feature
    node = create_node(feature=best_feature)

    # Iterate through each unique value of the best feature
    for v in unique_values(D, best_feature):
        # Create a subset of D where best_feature has the value v
        subset = create_subset(D, best_feature, v)

        if subset:
            # Recursively build the tree for the subset
            child_node = Build_Decision_Tree(subset, Features - {best_feature})
            add_child(node, child_node)
        else:
            # If subset is empty, create a leaf node with the most common class label
            leaf_node = create_leaf_node(class_label=most_common_class_label(D))
            add_child(node, leaf_node)

    return node
```
## Explanation
*   ### ``Inputs``
    *   ``D`` : The dataset containing the instances and their class labels.
    * ``Features`` : A list of features used to split the dataset.

    ## ``Process``
    ### 1. Check if all instances in D belong to the same class:
    * If true, a leaf node is created with that class label and returned.

    ### 2. Check if the feature list is empty:
    * If there are no more features left to split on, a leaf node with the most common class label in D is created and returned.

    ### 3. Select the best feature based on information gain:
    *  The function calculates the best feature by selecting the one with the highest information gain.

    ### 4.Split the dataset:
    * For each unique value v of the best feature:
        * A subset of the dataset is created where best_feature = v.
        * If the subset is non-empty, a child node is created by recursively calling Build_Decision_Tree on the subset and remaining features.
        * If the subset is empty, a leaf node with the most common class label in D is created.
