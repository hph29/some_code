package CodingQuestions;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val){
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

public class ValidateBST {

    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isValid(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    public boolean isValid(TreeNode node, Integer left, Integer right){
        if (node == null){
            return true;
        }
        if (left >= node.val || right <= node.val) {
            return false;
        }
        return isValid(node.right, node.val, right) && isValid(node.left, left, node.val);
    }
}
