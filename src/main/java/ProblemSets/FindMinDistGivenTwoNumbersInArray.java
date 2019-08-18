package ProblemSets;

public class FindMinDistGivenTwoNumbersInArray {

    /*
You are given an array A, of N elements. You need to find minimum distance between given two integers x and y.

Distance: The distance (index-based) between two elements of the array.
 */
    public static void main(String[] args) {
        GfG g = new GfG();
        Long[] arr = new Long[]{1L, 2L, 3L, 2L, 4L, 5L};
        System.out.println(g.minDist(arr, 5, 2, 4));
    }
}
/*This is a function problem.You only need to complete the function given below*/
/*Complete the function below*/
class GfG {
    long minDist(Long[] arr, long n, long x, long y) {
        int min_dist = arr.length;
        Long ref = Long.MAX_VALUE;
        int index_ref = -1;
        // add code here.
        for (int i = 0; i < n; i++) {
            if (ref != Long.MAX_VALUE && (arr[i] != ref && (arr[i] == x || arr[i] == y))) {
                min_dist = ((i - index_ref) < min_dist) ? (i - index_ref) : min_dist;
                ref = arr[i];
                index_ref = i;
            } else if (arr[i] == x || arr[i] == y) {
                ref = arr[i];
                index_ref = i;
            }
        }
        min_dist = (min_dist == arr.length) ? -1 : min_dist;
        return min_dist;
    }
}
