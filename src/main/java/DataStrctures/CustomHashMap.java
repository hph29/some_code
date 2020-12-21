package DataStrctures;

import java.util.LinkedList;
import java.util.Objects;

class Tuple2<K, V>{
    public final K key;
    public final V value;
    Tuple2(K key, V value){
        this.key = key;
        this.value = value;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tuple2<?, ?> tuple2 = (Tuple2<?, ?>) o;
        return tuple2.key == this.key && tuple2.value == this.value;
    }

    @Override
    public int hashCode() {
        return Objects.hash(key, value);
    }
}

public class CustomHashMap<K, V> {
    int size;
    LinkedList<Tuple2<K, V>>[] data;

    CustomHashMap(){
        this.size = 5;
        this.data = new LinkedList[this.size];
        for (int i=0; i<this.data.length; i++){
            this.data[i] = new LinkedList<>();
        }
    }
    CustomHashMap(int size){
        this.size = size;
        this.data = new LinkedList[this.size];
        for (int i=0; i<this.data.length; i++){
            this.data[i] = new LinkedList<>();
        }
    }
    public void put(K k, V v){
        int address = k.hashCode() % this.size;
        LinkedList<Tuple2<K, V>> selectedList = this.data[address];
        Tuple2<K, V> tupleToBeAdded = new Tuple2<>(k, v);
        if (selectedList.indexOf(tupleToBeAdded) == -1){
            selectedList.add(tupleToBeAdded);
        }
    }
    public V getFromKey(K k){
        int address = k.hashCode() % this.size;
        LinkedList<Tuple2<K, V>> selectedList = this.data[address];
        for (int i =0; i < selectedList.size(); i++){
            if (selectedList.get(i).key == k){
                return selectedList.get(i).value;
            }
        }
        return null;
    }

    public LinkedList<K> keys(){
        LinkedList<K> keys = new LinkedList<>();
        for(int i=0; i<this.size; i++){
            if (this.data[i].size() != 0){
                this.data[i].forEach(kvTuple2 -> keys.add(kvTuple2.key));
            }
        }
        return keys;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i=0; i< this.size; i++){
            if (this.data[i].size() != 0){
                sb.append(String.format("address %s:", i));
                this.data[i].forEach(kvTuple2 -> sb.append(String.format("(%s:%s)", kvTuple2.key, kvTuple2.value)));
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    public static void main(String[] args){
        CustomHashMap<String, Integer> customHashMap = new CustomHashMap<>();
        customHashMap.put("a", 1);
        customHashMap.put("b", 2);
        customHashMap.put("c", 3);
        customHashMap.put("d", 4);
        customHashMap.put("e", 5);
        customHashMap.put("f", 5);
        System.out.println(customHashMap.toString());
        System.out.println(customHashMap.getFromKey("a"));
        System.out.println(customHashMap.keys());
    }


}
