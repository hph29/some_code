1. Given a non-empty 2D array  `grid`  of 0's and 1's, an  **island**  is a group of  `1`'s (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
* Find the maximum area of an island in the given 2D array. (If there is no island, the maximum area is 0.)
```java
class Solution {
    int[][] grid;
    boolean[][] seen;

    public int area(int r, int c) {
        if (r < 0 || r >= grid.length || c < 0 || c >= grid[0].length ||
                seen[r][c] || grid[r][c] == 0)
            return 0;
        seen[r][c] = true;
        return (1 + area(r+1, c) + area(r-1, c)
                  + area(r, c-1) + area(r, c+1));
    }

    public int maxAreaOfIsland(int[][] grid) {
        this.grid = grid;
        seen = new boolean[grid.length][grid[0].length];
        int ans = 0;
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                ans = Math.max(ans, area(r, c));
            }
        }
        return ans;
    }
} //mark seen as '0' instead of explictly recording it.
```
-   Time Complexity:  O(R*C)O(R∗C), where  RR  is the number of rows in the given  `grid`, and  CC  is the number of columns. We visit every square once.
-   Space complexity:  O(R*C)O(R∗C), the space used by  `seen`  to keep track of visited squares, and the space used by the call stack during our recursion.
2. Design a data structure that supports insert, delete, search and getRandom in constant time.
	_**insert(x)**_  
	1) Check if x is already present by doing a hash map lookup.  
	2) If not present, then insert it at the end of the array.  
	3) Add in hash table also, x is added as key and last array index as index.

	_**remove(x)**_  
	1) Check if x is present by doing a hash map lookup.  
	2) If present, then find its index and remove it from hash map.  
	3) Swap the last element with this element in array and remove the last element.  
	Swapping is done because the last element can be removed in O(1) time.  
	4) Update index of last element in hash map.

	_**getRandom()**_  
	1) Generate a random number from 0 to last index.  
	2) Return the array element at the randomly generated index.

	_**search(x)**_  
	Do a lookup for x in hash map.
3. Given a BST, transform it into greater sum tree where each node contains sum of all nodes greater than that node.
	**Method 1 (Naive):**  
	This method doesn’t require the tree to be a BST. Following are steps.  
	1. Traverse node by node(Inorder, preorder, etc.)  
	2. For each node find all the nodes greater than that of the current node, sum the values. Store all these sums.  
	3. Replace each node value with their corresponding sum by traversing in the same order as in Step 1.  
	This takes O(n^2) Time Complexity.

	**Method 2 (Using only one traversal)**  
	By leveraging the fact that the tree is a BST, we can find an O(n) solution. The idea is to traverse BST in reverse inorder. Reverse inorder traversal of a BST gives us keys in decreasing order. Before visiting a node, we visit all greater nodes of that node. While traversing we keep track of sum of keys which is the sum of all the keys greater than the key of current node.
```c++
// C++ program to transform a BST to sum tree
#include<iostream>
using namespace std;
 
// A BST node
struct Node
{
    int data;
    struct Node *left, *right;
};
 
// A utility function to create a new Binary Tree Node
struct Node *newNode(int item)
{
    struct Node *temp =  new Node;
    temp->data = item;
    temp->left = temp->right = NULL;
    return temp;
}
 
// Recursive function to transform a BST to sum tree.
// This function traverses the tree in reverse inorder so
// that we have visited all greater key nodes of the currently
// visited node
void transformTreeUtil(struct Node *root, int *sum)
{
   // Base case
   if (root == NULL)  return;
 
   // Recur for right subtree
   transformTreeUtil(root->right, sum);
 
   // Update sum
   *sum = *sum + root->data;
 
   // Store old sum in current node
   root->data = *sum - root->data;
 
   // Recur for left subtree
   transformTreeUtil(root->left, sum);
}
 
// A wrapper over transformTreeUtil()
void transformTree(struct Node *root)
{
    int sum = 0; // Initialize sum
    transformTreeUtil(root, &sum);
}
 
// A utility function to print indorder traversal of a
// binary tree
void printInorder(struct Node *root)
{
    if (root == NULL) return;
 
    printInorder(root->left);
    cout << root->data << " ";
    printInorder(root->right);
}
 
// Driver Program to test above functions
int main()
{
    struct Node *root = newNode(11);
    root->left = newNode(2);
    root->right = newNode(29);
    root->left->left = newNode(1);
    root->left->right = newNode(7);
    root->right->left = newNode(15);
    root->right->right = newNode(40);
    root->right->right->left = newNode(35);
 
    cout << "Inorder Traversal of given tree\n";
    printInorder(root);
 
    transformTree(root);
 
    cout << "\n\nInorder Traversal of transformed tree\n";
    printInorder(root);
 
    return 0;
}
```
4. Merge Sort
```java
/* Java program for Merge Sort */
class MergeSort
{
    // Merges two subarrays of arr[].
    // First subarray is arr[l..m]
    // Second subarray is arr[m+1..r]
    void merge(int arr[], int l, int m, int r)
    {
        // Find sizes of two subarrays to be merged
        int n1 = m - l + 1;
        int n2 = r - m;
 
        /* Create temp arrays */
        int L[] = new int [n1];
        int R[] = new int [n2];
 
        /*Copy data to temp arrays*/
        for (int i=0; i<n1; ++i)
            L[i] = arr[l + i];
        for (int j=0; j<n2; ++j)
            R[j] = arr[m + 1+ j];
 
 
        /* Merge the temp arrays */
 
        // Initial indexes of first and second subarrays
        int i = 0, j = 0;
 
        // Initial index of merged subarry array
        int k = l;
        while (i < n1 && j < n2)
        {
            if (L[i] <= R[j])
            {
                arr[k] = L[i];
                i++;
            }
            else
            {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
 
        /* Copy remaining elements of L[] if any */
        while (i < n1)
        {
            arr[k] = L[i];
            i++;
            k++;
        }
 
        /* Copy remaining elements of R[] if any */
        while (j < n2)
        {
            arr[k] = R[j];
            j++;
            k++;
        }
    }
 
    // Main function that sorts arr[l..r] using
    // merge()
    void sort(int arr[], int l, int r)
    {
        if (l < r)
        {
            // Find the middle point
            int m = (l+r)/2;
 
            // Sort first and second halves
            sort(arr, l, m);
            sort(arr , m+1, r);
 
            // Merge the sorted halves
            merge(arr, l, m, r);
        }
    }
 
    /* A utility function to print array of size n */
    static void printArray(int arr[])
    {
        int n = arr.length;
        for (int i=0; i<n; ++i)
            System.out.print(arr[i] + " ");
        System.out.println();
    }
 
    // Driver method
    public static void main(String args[])
    {
        int arr[] = {12, 11, 13, 5, 6, 7};
 
        System.out.println("Given Array");
        printArray(arr);
 
        MergeSort ob = new MergeSort();
        ob.sort(arr, 0, arr.length-1);
 
        System.out.println("\nSorted array");
        printArray(arr);
    }
}
/* This code is contributed by Rajat Mishra */
```
5. Quick Sort
```java
// Java program for implementation of QuickSort
class QuickSort
{
    /* This function takes last element as pivot,
       places the pivot element at its correct
       position in sorted array, and places all
       smaller (smaller than pivot) to left of
       pivot and all greater elements to right
       of pivot */
    int partition(int arr[], int low, int high)
    {
        int pivot = arr[high]; 
        int i = (low-1); // index of smaller element
        for (int j=low; j<high; j++)
        {
            // If current element is smaller than or
            // equal to pivot
            if (arr[j] <= pivot)
            {
                i++;
 
                // swap arr[i] and arr[j]
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
 
        // swap arr[i+1] and arr[high] (or pivot)
        int temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;
 
        return i+1;
    }
 
 
    /* The main function that implements QuickSort()
      arr[] --> Array to be sorted,
      low  --> Starting index,
      high  --> Ending index */
    void sort(int arr[], int low, int high)
    {
        if (low < high)
        {
            /* pi is partitioning index, arr[pi] is 
              now at right place */
            int pi = partition(arr, low, high);
 
            // Recursively sort elements before
            // partition and after partition
            sort(arr, low, pi-1);
            sort(arr, pi+1, high);
        }
    }
 
    /* A utility function to print array of size n */
    static void printArray(int arr[])
    {
        int n = arr.length;
        for (int i=0; i<n; ++i)
            System.out.print(arr[i]+" ");
        System.out.println();
    }
 
    // Driver program
    public static void main(String args[])
    {
        int arr[] = {10, 7, 8, 9, 1, 5};
        int n = arr.length;
 
        QuickSort ob = new QuickSort();
        ob.sort(arr, 0, n-1);
 
        System.out.println("sorted array");
        printArray(arr);
    }
}
/*This code is contributed by Rajat Mishra */
```
6. Binary Search
```java
// Java implementation of recursive Binary Search
class BinarySearch
{
    // Returns index of x if it is present in arr[l..
    // r], else return -1
    int binarySearch(int arr[], int l, int r, int x)
    {
        if (r>=l)
        {
            int mid = l + (r - l)/2;
 
            // If the element is present at the 
            // middle itself
            if (arr[mid] == x)
               return mid;
 
            // If element is smaller than mid, then 
            // it can only be present in left subarray
            if (arr[mid] > x)
               return binarySearch(arr, l, mid-1, x);
 
            // Else the element can only be present
            // in right subarray
            return binarySearch(arr, mid+1, r, x);
        }
 
        // We reach here when element is not present
        //  in array
        return -1;
    }
 
    // Driver method to test above
    public static void main(String args[])
    {
        BinarySearch ob = new BinarySearch();
        int arr[] = {2,3,4,10,40};
        int n = arr.length;
        int x = 10;
        int result = ob.binarySearch(arr,0,n-1,x);
        if (result == -1)
            System.out.println("Element not present");
        else
            System.out.println("Element found at index " + 
                                                 result);
    }
}
/* This code is contributed by Rajat Mishra */
```

7. Binary Heap (Priority Queue)
```java
class MinHeap
{
	private int[] Heap;
    private int size;
    private int maxsize;

    public MinHeap(int maxsize)
    {
        this.maxsize = maxsize;
        this.size = 0;
        Heap = new int[this.maxsize];
        // Heap[0] = Integer.MIN_VALUE;
    }

    private int parent(int pos)
    {
        return (pos-1) / 2;
    }
 
    private int left(int pos)
    {
        return (2 * pos + 1);
    }
 
    private int right(int pos)
    {
        return (2 * pos) + 2;
    }

    private void swap(int fpos, int spos)
    {
        int tmp;
        tmp = Heap[fpos];
        Heap[fpos] = Heap[spos];
        Heap[spos] = tmp;
    }

    // Inserts a new key 'k'
	public void insertKey(int k)
	{
	    if (size == maxsize)
	    {
	        System.out.println("Overflow: Could not insertKey");
	        return;
	    }
	 
	    // First insert the new key at the end
	    size++;
	    int i = size - 1;
	    Heap[i] = k;
	 
	    // Fix the min heap property if it is violated
	    while (i != 0 && Heap[parent(i)] > Heap[i])
	    {
	       swap(i, parent(i));
	       i = parent(i);
	    }
	}

	// Decreases value of key at index 'i' to new_val.  It is assumed that
	// new_val is smaller than harr[i].
	public void decreaseKey(int i, int new_val)
	{
	    Heap[i] = new_val;
	    while (i != 0 && Heap[parent(i)] > Heap[i])
	    {
	       swap(i, parent(i));
	       i = parent(i);
	    }
	}

	public int extractMin()
	{
	    if (size <= 0)
	        return Integer.MAX_VALUE;
	    if (size == 1)
	    {
	        size--;
	        return Heap[0];
	    }
	 
	    // Store the minimum value, and remove it from heap
	    int root = Heap[0];
	    Heap[0] = Heap[size-1];
	    size--;
	    MinHeapify(0);
	 
	    return root;
	}

	// This function deletes key at index i. It first reduced value to minus
	// infinite, then calls extractMin()
	public void deleteKey(int i)
	{
	    decreaseKey(i, Integer.MIN_VALUE);
	    extractMin();
	}

	// A recursive method to heapify a subtree with the root at given index
	// This method assumes that the subtrees are already heapified
	public void MinHeapify(int i)
	{
	    int l = left(i);
	    int r = right(i);
	    int smallest = i;
	    if (l < size && Heap[l] < Heap[i])
	        smallest = l;
	    if (r < size && Heap[r] < Heap[smallest])
	        smallest = r;
	    if (smallest != i)
	    {
	        swap(i, smallest);
	        MinHeapify(smallest);
	    }
	}

	public void print()
    {
        for (int i = 0; i < size / 2; i++ )
        {
            System.out.print(" PARENT : " + Heap[i] + " LEFT CHILD : " + Heap[2*i + 1] 
                + " RIGHT CHILD :" + Heap[2 * i  + 2]);
            System.out.println();
        } 
    }

	public static void main(String[] arg)
    {
        System.out.println("The Min Heap is ");
        MinHeap minHeap = new MinHeap(15);
        minHeap.insertKey(5);
        minHeap.insertKey(3);
        minHeap.insertKey(17);
        minHeap.insertKey(10);
        minHeap.insertKey(84);
        minHeap.insertKey(19);
        minHeap.insertKey(6);
        minHeap.insertKey(22);
        minHeap.insertKey(9);
 
        minHeap.print();
        System.out.println("The Min val is " + minHeap.extractMin());
    }
}
```