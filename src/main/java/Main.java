/**
 * Created by hph on 2018-07-15.
 */
public class Main {

    public static void main(String[] args) {
        Tree tree = new Tree().createBalancedTree();

        printSection("### Row level print");
        tree.rowLevelPrint();

        printSection("### Pretty Print");
        tree.prettyPrint();

        printSection("### Get Layer Num Test");
        System.out.print(tree.getNumLayer());


        printSection("### Search Test");
        System.out.println(tree.search(20));
        System.out.println(tree.search(100));

        printSection("### Deletion Test");
        tree.delete(20);
        tree.print();
        tree.delete(40);
        tree.delete(100);
        tree.print();
    }

    static void printSection(String header){
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println(header);
    }
}
