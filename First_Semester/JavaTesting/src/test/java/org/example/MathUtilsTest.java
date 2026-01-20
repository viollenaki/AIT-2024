package org.example;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MathUtilsTest {

    @Test
    void factorial() {
        assertEquals(1, MathUtils.factorial(0));
        assertEquals(1, MathUtils.factorial(1));
        assertEquals(120, MathUtils.factorial(5));
        assertEquals(2432902008176640000L, MathUtils.factorial(20));
        assertThrows(IllegalArgumentException.class, () -> MathUtils.factorial(-1));
    }

    @Test
    void isPrime() {
        assertFalse(MathUtils.isPrime(0));
        assertFalse(MathUtils.isPrime(1));
        assertTrue(MathUtils.isPrime(2));
        assertTrue(MathUtils.isPrime(3));
        assertTrue(MathUtils.isPrime(17));
        assertFalse(MathUtils.isPrime(18));
        assertFalse(MathUtils.isPrime(-5));
    }

    @Test
    void average() {
        assertEquals(2.5, MathUtils.average(1, 2, 3, 4), 1e-9);
        assertEquals(5.0, MathUtils.average(5.0), 1e-9);
        assertThrows(IllegalArgumentException.class, () -> MathUtils.average());
        assertThrows(IllegalArgumentException.class, () -> MathUtils.average((double[]) null));
    }

    @Test
    void absDiff() {
        assertEquals(2.0, MathUtils.absDiff(5, 3), 1e-9);
        assertEquals(2.0, MathUtils.absDiff(3, 5), 1e-9);
        assertEquals(0.0, MathUtils.absDiff(2.5, 2.5), 1e-9);
    }
}