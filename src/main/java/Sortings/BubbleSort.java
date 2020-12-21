package Sortings;

import java.util.Arrays;

public class BubbleSort {

    static void swap(int[] arr, int i, int j){
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    static int[] bubbleSort(int[] arr){
        for(int i = 0; i < arr.length - 1; i++){
            for(int j = i + 1; j < arr.length; j++){
                if (arr[i] > arr[j]){
                    swap(arr, i, j);
                }
            }
        }
        return arr;
    }

    public static void main(String[] args){
      int[] arr = new int[] {9,8,7,1,2,3,6,5,4};
      int[] newArr = bubbleSort(arr);
      Arrays.stream(newArr).forEach(value -> System.out.print(value + " "));
    }
}
