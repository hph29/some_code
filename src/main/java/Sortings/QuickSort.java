package Sortings;

import java.util.Arrays;

public class QuickSort {


    static void swap(int[] arr, int i, int j){
        System.out.println("swapping" + " " + arr[i] + " " + arr[j]);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    static void quickSort(int[] arr, int low, int high){
        if (low < high){
            int index = partition(arr, low, high);

            quickSort(arr, low, index-1);
            quickSort(arr, index+1, high);
        }
    }

    static int partition(int[] arr, int low, int high){
        int pivotValue = arr[high];
        int partitionIndex = low;
        for(int i=low; i < high; i++){
            if (arr[i] < pivotValue){
                swap(arr, i, partitionIndex);
                partitionIndex++;
            }
        }
        swap(arr, high, partitionIndex);
        return partitionIndex;
    }

    public static void main(String[] args){
      int[] arr = new int[] {9,8,7,1,2,3,6,5,4};
      quickSort(arr,0, arr.length-1);
      Arrays.stream(arr).forEach(value -> System.out.print(value + " "));
    }
}
