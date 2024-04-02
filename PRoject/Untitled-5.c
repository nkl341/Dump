#include <avr/io.h>
#include <avr/interrupt.h>

#define F_CPU 1000000UL  // Define CPU frequency as 1MHz
#include <util/delay.h> // Include AVR delay header

void delay_ms(uint16_t ms) {
    while (ms--) {
        _delay_ms(1); // Use _delay_ms from AVR LibC to delay 1 millisecond
    }
}

int main(void) {
    // Set PB0 as input (logic level selection)
    DDRB &= ~(1 << DDB0);
    // Enable pull-up resistor for PB0
    PORTB |= (1 << PORTB0);

    // Set PC0 as output
    DDRC |= (1 << DDC0);

    // Infinite loop
    while (1) {
        // Check the input signal at PB0
        if (PINB & (1 << PINB0)) {
            // PB0 is logic high, output 66% duty cycle
            PORTC |= (1 << PORTC0);  // Set PC0 high
            _delay_ms(660);  // Wait for 660ms
            PORTC &= ~(1 << PORTC0);  // Set PC0 low
            _delay_ms(340);  // Wait for 340ms
        } else {
            // PB0 is logic low, output 50% duty cycle
            PORTC |= (1 << PORTC0);  // Set PC0 high
            _delay_ms(500);  // Wait for 500ms
            PORTC &= ~(1 << PORTC0);  // Set PC0 low
            _delay_ms(500);  // Wait for 500ms
        }
    }

    return 0;
}
