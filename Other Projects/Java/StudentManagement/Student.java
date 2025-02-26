public class Student {
    private String name; // Student name
    private int age; // Student age
    private String id; // Student unique ID

    // Constructor: Initializes the student object
    public Student(String name, int age, String id) {
        this.name = name;
        this.age = age;
        this.id = id;
    }

    // ✅ Getter for student ID
    public String getId() {
        return id;
    }

    // ✅ Getter for student name
    public String getName() {
        return name;
    }

    // ✅ Getter for student age
    public int getAge() {
        return age;
    }

    // ✅ Converts Student object to a readable format (used in displayStudents)
    @Override
    public String toString() {
        return "ID: " + id + " | Name: " + name + " | Age: " + age;
    }
}
