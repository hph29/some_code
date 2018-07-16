import java.util.LinkedList;

/**
 * Created by hph on 2018-07-15.
 */
public class Tree {

    class Node{
        int value;
        Node left;
        Node right;

        Node(int value){
            this.value = value;
            left = right = null;
        }
    }

    Node root = null;

    Tree(){}
    Tree(int value){
        this.root = new Node(value);
    }

    void insert(int value){
        // important to assign root to the result node
        root = insertRec(root, value);
    }

    // insert func returns the node after the value got inserted
    private Node insertRec(Node node, int value){
        if (node == null){
            node = new Node(value);
        }
        else if (node.value > value){
            node.left = insertRec(node.left, value);
        }
        else if (node.value < value){
            node.right = insertRec(node.right, value);
        }
        return node;
    }

    void print(){
        System.out.println("Print Tree");
        print(root);
        System.out.println();
        System.out.println("End of Print Tree");
    }

    private void print(Node node){
        if (node != null) {
            print(node.left);
            System.out.print(node.value + " ");
            print(node.right);
        }
    }

    boolean search(int value){
        return search(root, value);
    }

    private boolean search(Node node, int value){
        if (node == null){
            return false;
        }
        if (node.value == value){
            return true;
        }
        else if (node.value > value){
            return search(node.left, value);
        }
        else if (node.value < value) {
            return search(node.right, value);
        }
        return false;
    }

    void delete(int value){
        delete(root, value);
    }

    // 1. traverse tree to find node
    // 2. once find, 3 case:
    //    i.   no child, make null
    //    ii.  1 child, make curr node to the child
    //    iii. 2 child, find the right of smallest value
    //         once find, make right child to that node and delete the previous found node.
    // Note: delete func needs to return the entire tree after deleting the value.
    private Node delete(Node node, int value){
        if (node == null){return null;}
        else{
            if (node.value > value){
                node.left = delete(node.left, value);
            }
            else if (node.value < value){
                node.right = delete(node.right, value);
            }
            else {
                if (node.left == null && node.right == null){
                    node = null;
                }
                else if (node.left == null){
                    node = node.right;
                }
                else if (node.right == null){
                    node = node.left;
                }
                else {
                    node.value = findRightSmallestValue(node.right);
                    node.right = delete(node.right, node.value);
                }
            }
            return node;
        }
    }

    private int findRightSmallestValue(Node node){
        int v = node.value;
        while(node.left != null){
            v = node.left.value;
            node = node.left;
        }
        return v;
    }

    int getNumLayer(){
        return getNumLayer(root, 0);
    }

    private int getNumLayer(Node node, int acc){
        if (node != null){
            acc += 1;
            acc = Math.max(getNumLayer(node.left, acc), getNumLayer(node.right, acc));
        }
        return acc;
    }


    void rowLevelPrint(){
        for (int i = 1; i <= getNumLayer(); i ++){
            rowLevelPrint(root, i);
            System.out.println();
        }

    }

    // print nth layer's element in the tree
    // since it only print one layer per call, need super function call this func for each layer.
    private void rowLevelPrint(Node node , int level)
    {
        if (node == null)
            return;
        if (level == 1)
            System.out.print(node.value + " ");
        else if (level > 1)
        {
            rowLevelPrint(node.left, level-1);
            rowLevelPrint(node.right, level-1);
        }
    }

    void prettyPrint(){
        if (root == null) return;
        for (int i = 1; i <= getNumLayer(); i++){
            prettyPrint(root, getNumLayer(), 1, new LinkedList<Integer>(), i);
            System.out.println();
        }
    }

    // Utilized row level print tree mechanism, for each row, calculate indent and space and print value.
    // In order to get indent and space, max layer of the tree and current layer of the tree are needed.
    // In order to calculate current position of the given row, need to record the parent path of the tree.
    @SuppressWarnings("unchecked")
    private void prettyPrint(Node node, int maxLayer, int currLayer, LinkedList<Integer> parent, int lvl){
        if (node != null){
            if (lvl == 1){
                int indent = powerOf(2, maxLayer-currLayer) - 1;
                int space = powerOf(2, maxLayer-currLayer + 1) - 1;
                StringBuilder sb = new StringBuilder();

                for (int i = 0; i < indent + getPos(parent) * space; i++){
                    sb.append(" ");
                }
                sb.append(node.value);
                System.out.print(sb.toString());
            }
            else if (lvl > 1){
                LinkedList<Integer> leftParent = (LinkedList<Integer>) parent.clone();
                leftParent.addLast(0);
                prettyPrint(node.left, maxLayer, currLayer+1, leftParent, lvl - 1);
                LinkedList<Integer> rightParent = (LinkedList<Integer>) parent.clone();
                rightParent.addLast(1);
                prettyPrint(node.right, maxLayer, currLayer+1, rightParent, lvl - 1);
            }
        }
    }

    // Calculate the current position by given the parent path
    // e.g. {left, right, right} -> {0, 1, 1} -> 3
    // e.g. {left, left, left} -> {0, 0, 0} -> 0
    // e.g. {right, right, right} -> {1, 1, 1} -> 1 * 2**(3-1) +1 * 2**(2-1) +1 * 2**(1-1) = 7 last element in 3 layer binary tree
    private int getPos(LinkedList<Integer> list){
        int pos = 0;
        for (int i = 0; i < list.size(); i++){
            pos += list.get(i) * powerOf(2, (list.size() - i - 1));
        }
        return pos;
    }

    private int powerOf(int a, int b){
        double d_a = (double) a;
        double d_b = (double) b;
        return (int) Math.pow(d_a, d_b);
    }

    Tree createBalancedTree(){
        Tree tree = new Tree();
        Node node1 = new Node(10);
        Node node2 = new Node(20);
        Node node3 = new Node(30);
        Node node4 = new Node(40);
        Node node5 = new Node(50);
        Node node6 = new Node(60);
        Node node7 = new Node(70);
        tree.root = node4;
        node4.left = node2;
        node4.right = node6;
        node2.left = node1;
        node2.right = node3;
        node6.left = node5;
        node6.right = node7;
        return tree;
    }


}
