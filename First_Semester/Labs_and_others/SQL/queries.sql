CREATE TABLE patients (
    patient_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    age INT,
    weight DECIMAL(5,2),
    allergies VARCHAR(100),
    city VARCHAR(50)
);

INSERT INTO patients (patient_id, first_name, last_name, age, weight, allergies, city)
VALUES
(1, 'John', 'Smith', 30, 110.5, NULL, 'Bishkek'),
(2, 'Cathy', 'Brown', 25, 105.0, 'Peanuts', 'Osh'),
(3, 'Chris', 'White', 40, 120.0, NULL, 'Bishkek'),
(4, 'Anna', 'Green', 35, 98.0, 'Dust', 'Karakol'),
(5, 'David', 'Black', 50, 130.0, 'Pollen', 'Tokmok');

SELECT * FROM patients



INSERT INTO patients VALUES (6, 'Carl', 'Young', 28, 115.0, NULL, 'Naryn');


SELECT * from patients
WHERE city = "Bishkek"

SELECT * FROM patients
where age < 30;

SELECT first_name, last_name From patients
where first_name like "c%"

SELECT first_name, last_name From patients
where last_name like "%n"

SELECT first_name, last_name, city from patients
where first_name like "%o%" OR last_name like "%o%" or city like "%o%"

SELECT first_na


