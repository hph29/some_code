package CodingQuestions;

public class MergedSortedArray {
    public static int[] mergeSortedArray(int[] array1, int[] array2){
        int[] result = new int[array1.length + array2.length];
        int index_1 = 0;
        int index_2 = 0;
        while (index_1 < array1.length || index_2 < array2.length){
            if (index_2 == array2.length || array1[index_1] < array2[index_2]){
                result[index_1+index_2] = array1[index_1];
                index_1++;
            }
            else{
                result[index_1+index_2] = array2[index_2];
                index_2++;
            }
        }
        return result;
    }

    public static void main(String[] args){
        int[] array1 = new int[] {1,3,5,7};
        int[] array2 = new int[] {2,4,6,7};
        int[] sortedArray =  mergeSortedArray(array1, array2);
        for (int i =0; i < sortedArray.length; i++){
            System.out.print(sortedArray[i]);
            System.out.println();
        }


    }
}
