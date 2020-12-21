package Sortings;

import java.util.Arrays;

public class InsertionSort {

    // No swap should be used;

    static int[] insertionSort(int[] arr){

        for (int i= 1; i<arr.length; i++){
            int value = arr[i];
            int j = i - 1;
            while(j >= 0 && value < arr[j]){
                arr[j+1] = arr[j];
                j--;
            }
            arr[j+1] = value;
        }
        return arr;
    }

    public static void main(String[] args){
      int[] arr = new int[] {9,8,7,1,2,3,6,5,4};
      int[] newArr = insertionSort(arr);
      Arrays.stream(newArr).forEach(value -> System.out.print(value + " "));
    }
}
