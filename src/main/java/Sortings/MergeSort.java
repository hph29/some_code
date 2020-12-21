package Sortings;

import java.util.Arrays;

public class MergeSort {

    // No swap should be used;

    static int[] mergeSort(int[] arr){
        System.out.println(Arrays.toString(arr));
        if (arr.length == 1){
            return arr;
        }
        int mid = arr.length / 2;
        int[] left = Arrays.copyOfRange(arr, 0, mid);
        int[] right = Arrays.copyOfRange(arr, mid, arr.length);
        return merge(mergeSort(left), mergeSort(right));
    }

    static int[] merge(int[] left, int[] right){
        int[] result = new int[left.length + right.length];
        int i = 0;
        int j = 0;

        while(i != left.length || j != right.length){
            if (i == left.length){
                result[i+j] = right[j];
                j++;
            }
            else if (j == right.length){
                result[i+j] = left[i];
                i++;
            }
            else{
                if (left[i] > right[j]){
                    result[i+j] = right[j];
                    j++;
                }
                else{
                    result[i+j] = left[i];
                    i++;
                }
            }
        }
        return result;
    }

    public static void main(String[] args){
      int[] arr = new int[] {9,8,7,1,2,3,6,5,4};
      int[] newArr = mergeSort(arr);
      Arrays.stream(newArr).forEach(value -> System.out.print(value + " "));
    }
}
