from pattern import Checker, Circle, Spectrum


def main():
    checker = Checker(100, 10)
    checker.draw()
    checker.show()

    circle = Circle(512, 50, (250, 60))
    circle.draw()
    circle.show()

    spectrum = Spectrum(100)
    spectrum.draw()
    spectrum.show()


if __name__ == '__main__':
    main()
