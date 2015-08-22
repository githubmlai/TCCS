from __future__ import division  # force floating point division
import model_generation as mg

if __name__ == '__main__':
    from pylab import *
    from matplotlib.widgets import Slider, Button, RadioButtons

    ax = subplot(111)
    subplots_adjust(left=0.25, bottom=0.50)
    support_start = 0.0
    support_stop = 1.0
    support_step = 0.001
    list_wave_number = arange(support_start, support_stop, support_step) 
    hurst_related_exponent_initial = 0.5
    characteristic_dist_initial = 3
    von_karman_energy_spectrum = mg.calc_von_karman_energy_spectrum(list_wave_number,
                                                                    hurst_related_exponent_initial,
                                                                    characteristic_dist_initial)
    l, = plot(list_wave_number, von_karman_energy_spectrum, lw=2, color='red')
    l_two, = plot(list_wave_number, von_karman_energy_spectrum, lw=2, color='blue')

    axis([support_start, support_stop, -10, 10])

    axcolor = 'lightgoldenrodyellow'
    axis_character_dist = axes([0.25, 0.10, 0.65, 0.03], axisbg=axcolor)
    axis_hurst_related_exp = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
    axis_character_dist_two = axes([0.25, 0.20, 0.65, 0.03], axisbg=axcolor)
    axis_hurst_related_exp_two = axes([0.25, 0.25, 0.65, 0.03], axisbg=axcolor)

    hurst_related_exp_min = -0.25
    hurst_related_exp_max = 1
    characteristic_dist_min = 0
    characteristic_dist_max = 10

    hurst_related_exp_slider = Slider(axis_hurst_related_exp,
                                      'H',
                                      hurst_related_exp_min,
                                      hurst_related_exp_max,
                                      valinit=hurst_related_exponent_initial)
    characteristic_dist_slider = Slider(axis_character_dist,
                                        'b',
                                        characteristic_dist_min,
                                        characteristic_dist_max,
                                        valinit=characteristic_dist_initial)
    hurst_related_exp_slider_two = Slider(axis_hurst_related_exp_two,
                                          'H',
                                          hurst_related_exp_min,
                                          hurst_related_exp_max,
                                          valinit=hurst_related_exponent_initial)
    characteristic_dist_slider_two = Slider(axis_character_dist_two,
                                            'b',
                                            characteristic_dist_min,
                                            characteristic_dist_max,
                                            valinit=characteristic_dist_initial)

    def update(val):
        hurst_related_exponent = hurst_related_exp_slider.val
        characteristic_dist = characteristic_dist_slider.val
        von_karman_energy_spectrum = mg.calc_von_karman_energy_spectrum(list_wave_number,
                                                                        hurst_related_exponent,
                                                                        characteristic_dist)
        l.set_ydata(von_karman_energy_spectrum)
        hurst_related_exponent_two = hurst_related_exp_slider_two.val
        characteristic_dist_two = characteristic_dist_slider_two.val
        von_karman_energy_spectrum_two = mg.calc_von_karman_energy_spectrum(list_wave_number,
                                                                            hurst_related_exponent_two,
                                                                            characteristic_dist_two)
        l_two.set_ydata(von_karman_energy_spectrum_two)
        draw()

    characteristic_dist_slider.on_changed(update)
    hurst_related_exp_slider.on_changed(update)
    characteristic_dist_slider_two.on_changed(update)
    hurst_related_exp_slider_two.on_changed(update)

    reset_axis = axes([0.8, 0.025, 0.1, 0.04])
    button = Button(reset_axis, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        characteristic_dist_slider.reset()
        hurst_related_exp_slider.reset()
        characteristic_dist_slider_two.reset()
        hurst_related_exp_slider_two.reset()
    button.on_clicked(reset)

    rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
    radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

    def colorfunc(label):
        l.set_color(label)
        draw()
    radio.on_clicked(colorfunc)

    show()