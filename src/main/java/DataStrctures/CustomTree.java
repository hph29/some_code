package DataStrctures;

import javax.xml.stream.FactoryConfigurationError;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Stack;

class TreeNode {
    int data;
    TreeNode left;
    TreeNode right;

    TreeNode(int value){
        this.data = value;
        this.left = null;
        this.right = null;
    }
}

public class CustomTree {

    TreeNode root;

    CustomTree(int value){
        this.root = new TreeNode(value);
    }

    public boolean remove(int value){
        TreeNode cur = this.root;
        TreeNode pre = null;
        while(cur != null){
            if (value < cur.data){
                pre = cur;
                cur = cur.left;
            }
            else if (value > cur.data){
                pre = cur;
                cur = cur.right;

            }
            else {
                // Found
                if (pre == null){
                    // only one node and it is to be removed.
                    this.root = null;
                }
                else if (cur.left == null && cur.right == null){
                    pre.right = null;
                    pre.left = null;
                }
                else if (cur.left != null && cur.right == null){
                    pre.left = cur.left;
                }
                else if (cur.right != null && cur.left == null){
                    pre.right = cur.right;
                }
                else{
                    // both right and left contains child
                    int a =1;
                }

            }
        }
        return false;
    }

    public boolean lookup(int value){
        TreeNode cur = this.root;
        while(cur != null){
            if (cur.data == value){
                return true;
            }
            else if (value < cur.data){
                cur = cur.left;
            }
            else{
                cur = cur.right;
            }
        }
        return false;
    }

    public void insert(int value){
        TreeNode node = new TreeNode(value);
        if (this.root == null){
            this.root = node;
        }
        TreeNode currNode = this.root;
        while(true){
            if(value < currNode.data){
                if(currNode.left == null){
                    currNode.left = node;
                    break;
                }
                else{
                    currNode = currNode.left;
                }
            }
            else{
                if(currNode.right == null){
                    currNode.right = node;
                    break;
                }
                else{
                    currNode = currNode.right;
                }
            }
        }
    }

    public void print(TreeNode node) {
        TreeNode curr = this.root;
        Stack<TreeNode> s = new Stack<>();
        while(curr != null) {
            System.out.println(curr.data);
            if (curr.left != null && curr.right != null){
                s.push(curr.right);
                curr = curr.left;
            }
            else if (curr.left != null){
                curr = curr.right;
            }
            else if (curr.right != null){
                curr = curr.left;
            }
            else if (curr.right == null && curr.left == null){
                if (s.empty()){
                    break;
                }
                else{
                    curr = s.pop();
                }
            }
        }
        //TODO unfinished :(
    }

    public List<Integer> breathFirstSearch(){
        ArrayList<TreeNode> queue = new ArrayList<>();
        queue.add(this.root);
        return breathFirstSearchRec(queue, new ArrayList<>());
    }

    public List<Integer> breathFirstSearchRec(List<TreeNode> queue, List<Integer> list){
        if (queue.isEmpty()){
            return list;
        }
        else{
            TreeNode treeNode = queue.remove(0);
            list.add(treeNode.data);
            if(treeNode.left != null){
                queue.add(treeNode.left);
            }
            if (treeNode.right != null){
                queue.add(treeNode.right);
            }
            return breathFirstSearchRec(queue, list);
        }
    }

    public static void main(String[] args){
        CustomTree t = new CustomTree(9);
        t.insert(4);
        t.insert(20);
        t.insert(1);
        t.insert(6);
        t.insert(15);
        t.insert(170);
//        t.print(t.root);
//        System.out.println(t.lookup(15));
//        System.out.println(t.lookup(16));
//        System.out.println(t.lookup(1));
        System.out.println(t.breathFirstSearch());
    }
}
