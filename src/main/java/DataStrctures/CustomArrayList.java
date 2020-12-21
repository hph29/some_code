package DataStrctures;

import java.util.ArrayList;

public class CustomArrayList<E> {

    ArrayList<E> data;
    int length;

    CustomArrayList(){
        this.data = new ArrayList<>();
        this.length = 0;
    }

    public E get(int index){
        return this.data.get(index);
    }

    public void push(E item){
        this.data.add(this.length, item);
        this.length++;
    }

    public E pop(){
        E lastItem = this.data.get(this.length-1);
        length--;
        return lastItem;
    }

    void shiftLeft(int index){
        for (int i = index; i < length-1; i++) {
            this.data.set(i, this.data.get(i+1));
        }
        this.data.remove(length - 1);
        this.length--;
    }

    public E delete(int index){
        E itemToBeRemoved = this.data.get(index);
        shiftLeft(index);
        return itemToBeRemoved;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i < this.length; i++){
            sb.append(this.data.get(i));
            sb.append(' ');
        }
        return sb.toString();
    }

    public static void main(String[] args){
        CustomArrayList<Integer> arrayList = new CustomArrayList<>();
        arrayList.push(1);
        arrayList.push(2);
        arrayList.push(3);
        arrayList.push(4);
        assert arrayList.toString() == "1 2 3 4 ";
        arrayList.delete(1);
        assert arrayList.toString() == "1 3 4 ";
        arrayList.push(5);
        assert arrayList.toString() == "1 3 4 5 ";
        int popItem = arrayList.pop();
        assert popItem == 5;
        assert arrayList.toString() == "1 3 4 ";
        System.out.println(arrayList.toString());
    }
}
