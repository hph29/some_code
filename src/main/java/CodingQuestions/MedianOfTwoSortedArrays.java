package CodingQuestions;
/*
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

Follow up: The overall run time complexity should be O(log (m+n)).

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
Example 3:

Input: nums1 = [0,0], nums2 = [0,0]
Output: 0.00000
Example 4:

Input: nums1 = [], nums2 = [1]
Output: 1.00000
Example 5:

Input: nums1 = [2], nums2 = []
Output: 2.00000


Constraints:

nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-106 <= nums1[i], nums2[i] <= 106
 */
public class MedianOfTwoSortedArrays {
    public double calculateMedian(int[] array){
        int n = array.length;
        if (n % 2 == 0){
            return (array[n/2] + array[n/2-1]) / 2.0;
        }
        else{
            return array[n/2];
        }
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length == 0 && nums2.length == 0){
            return 0;
        }

        else if (nums1.length == 0) {
            return calculateMedian(nums2);
        }
        else if (nums2.length == 0) {
            return calculateMedian(nums1);
        }
        else{
            int m = nums1.length;
            int n = nums2.length;

            int mid = (m + n) / 2;

            int cur_num=0;
            int pre_num=0;

            int index1 = 0;
            int index2 = 0;

            while((index1+index2) <= mid){
                pre_num = cur_num;
                if (index1 == m){
                    cur_num = nums2[index2];
                    index2++;
                }
                else if (index2 == n){
                    cur_num = nums1[index1];
                    index1++;
                }
                else if (nums1[index1] > nums2[index2]){
                    cur_num = nums2[index2];
                    index2++;
                }
                else{
                    cur_num = nums1[index1];
                    index1++;
                }
            }
            if ((m+n) % 2 == 0){
                return calculateMedian(new int[]{pre_num, cur_num});
            }
            else{
                return calculateMedian(new int[]{cur_num});
            }
        }
    }
}
