/**
 * Created by hph on 2018-07-15.
 */
public class Main {

    public static void main(String[] args) {
        fibonacciRelated();
    }

    private static void sortRelated(){
        int[] array = new int[]{9,1,8,2,7,3,6,4,5};
        printSection("MergeSort");
        printArray(array);
        Sort.mergeSort(array, 0, array.length -1 );
        printArray(array);

        array = new int[]{9,1,8,2,7,3,6,4,5};
        printSection("QuickSort");
        printArray(array);
        Sort.quickSort(array, 0, array.length -1 );
        printArray(array);

        array = new int[]{9,1,8,2,7,3,6,4,5};
        printSection("Insertion Sort");
        printArray(array);
        Sort.insertionSort(array);
        printArray(array);

        array = new int[]{9,1,8,2,7,3,6,4,5};
        printSection("Bubble Sort");
        printArray(array);
        Sort.bubbleSort(array);
        printArray(array);

    }

    private static void treeRelated(){
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

    private static void printArray(int[] array){
        System.out.print("Array: ");
        for(int i=0; i< array.length; i++){
            System.out.print(array[i] + " ");
        }
        System.out.println();
    }

    private static void printSection(String header){
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println(header);
    }

    private static void fibonacciRelated(){
        assert Fibonacci.fibonacci1(0) == 0;
        assert Fibonacci.fibonacci1(1) == 1;
        assert Fibonacci.fibonacci1(2) == 1;
        assert Fibonacci.fibonacci1(3) == 2;
        assert Fibonacci.fibonacci1(5) == 5;

        assert Fibonacci.fibonacci2(0) == 0;
        assert Fibonacci.fibonacci2(1) == 1;
        assert Fibonacci.fibonacci2(2) == 1;
        assert Fibonacci.fibonacci2(3) == 2;
        assert Fibonacci.fibonacci2(5) == 5;

        assert Fibonacci.fibonacci3(0) == 0;
        assert Fibonacci.fibonacci3(1) == 1;
        assert Fibonacci.fibonacci3(2) == 1;
        assert Fibonacci.fibonacci3(3) == 2;
        assert Fibonacci.fibonacci3(5) == 5;
    }
}
