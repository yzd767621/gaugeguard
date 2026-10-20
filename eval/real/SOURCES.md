# Real photographs: sources and licences

## Tuning set (`images/`, `labels.json`)

All photographs are from Wikimedia Commons and are used under the licence shown. They were downscaled to
1000 px wide; no other changes. Values in `labels.json` were read by eye by the GaugeGuard authors.
Files under CC BY-SA are redistributed under the same licence.

| File | Original | Author | Licence |
|---|---|---|---|
| `wika_contact_0-4bar.jpg` | [Bourdon tube pressure gauge glass front](https://commons.wikimedia.org/wiki/File:Bourdon_tube_pressure_gauge_glass_front.jpg) | DStaiger | CC BY-SA 3.0 |
| `wika_regulator_0-4000psi.jpg` | [Gauge](https://commons.wikimedia.org/wiki/File:Gauge.jpg) | Hey Paul from State College, PA, USA | CC BY 2.0 |
| `wika_minicheck_0-15psi.jpg` | [Hyperbaric oxygen cell testing pot pressure gauge](https://commons.wikimedia.org/wiki/File:Hyperbaric_oxygen_cell_testing_pot_pressure_gauge_IMG_20201209_153847.jpg) | Pbsouthwood | CC BY-SA 4.0 |
| `maximator_0-2500bar.jpg` | [MAXIMATOR High Pressure Manometer 01a](https://commons.wikimedia.org/wiki/File:MAXIMATOR-High-Pressure-Manometer-01a.jpg) | CEphoto, Uwe Aranas | CC BY-SA 3.0 |
| `soderstroms_0-15kgcm2.jpg` | [Manometer 104026](https://commons.wikimedia.org/wiki/File:Manometer_104026.jpg) | Nord68 | CC BY-SA 4.0 |
| `astralpool_0-3bar.jpg` | [Manómetro 3 psi 1](https://commons.wikimedia.org/wiki/File:Man%C3%B3metro_3_psi_1.jpg) | Amfeli | CC BY 4.0 |
| `guru_0-60psi.jpg` | [Pressure Gauge](https://commons.wikimedia.org/wiki/File:Pressure_Gauge.jpg) | In Transit | CC BY-SA 4.0 |
| `usgauge_0-4000psi.jpg` | [Psidial](https://commons.wikimedia.org/wiki/File:Psidial.jpg) | leapingllamas | CC BY 2.0 |
| `vdo_0-10bar.jpg` | [VDO engine oil pressure gauge](https://commons.wikimedia.org/wiki/File:VDO_engine_oil_pressure_gauge.jpg) | R. Henrik Nilsson | CC BY 4.0 |
| `marsh_1900_0-200psi.jpg` | [Pressure gauge circa 1900](https://commons.wikimedia.org/wiki/File:Pressure_gauge_circa_1900.jpg) | Runner1928 | CC BY-SA 4.0 |
| `station_garage_distant.jpg` | [Lakewood-Wadsworth Station garage pressure gauge](https://commons.wikimedia.org/wiki/File:Lakewood-Wadsworth_Station_garage_pressure_gauge.jpg) | Xnatedawgx | CC BY-SA 4.0 |
| `train_air_supply_small.jpg` | [Pressure gauge of air supply pipe](https://commons.wikimedia.org/wiki/File:Pressure_gauge_of_air_supply_pipe.jpg) | Fumikas Sagisavas | CC0 |
| `train_brake_pipe_small.jpg` | [Pressure gauge of brake pipe](https://commons.wikimedia.org/wiki/File:Pressure_gauge_of_brake_pipe.jpg) | Fumikas Sagisavas | CC0 |
| `reclaimed_water_distant.jpg` | [Reclaimed water pressure gage](https://commons.wikimedia.org/wiki/File:Reclaimed_water_pressure_gage.jpg) | Mds08011 | CC BY 4.0 |

Licence texts: [CC0](https://creativecommons.org/publicdomain/zero/1.0/),
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
`eval/results/real_tuning_overlays.jpg` is a derivative of these photographs and carries the same attributions.

## Held-out test set (`test_images/`, `test_labels.json`)

Collected with `eval/real/crawl.py` (Wikimedia Commons category tree and Openverse; only CC0, public domain,
CC BY and CC BY-SA kept), filtered with `eval/real/candidates.py` (dial locator only), and labelled by eye
before the improved reader was run on them, except where a note in `test_labels.json` says otherwise.
Downscaled to at most 1280 px; no other changes.

| File | Original | Author | Licence |
|---|---|---|---|
| `t001.jpg` | [Glycerine-filled pressure gauge up to 4 MPa](https://commons.wikimedia.org/wiki/File:Glycerine-filled_pressure_gauge_up_to_4_MPa.JPG) | Deyan Levski | CC BY-SA 3.0 |
| `t006.jpg` | [Atlas Copco XAHS 347-pic4-Manometer](https://commons.wikimedia.org/wiki/File:Atlas_Copco_XAHS_347-pic4-Manometer.jpg) | Cjp24 | CC BY-SA 4.0 |
| `t008.jpg` | [Balaklava Naval museum complex Depth gauge IMG 1056 1725](https://commons.wikimedia.org/wiki/File:Balaklava_Naval_museum_complex_Depth_gauge_IMG_1056_1725.jpg) | Alexxx1979 | CC BY-SA 3.0 |
| `t011.jpg` | [2015-03-07 - Thinktank - Sasha Taylor - 138](https://commons.wikimedia.org/wiki/File:2015-03-07_-_Thinktank_-_Sasha_Taylor_-_138.JPG) | Sasha Taylor | CC BY-SA 4.0 |
| `t020.jpg` | [Ĉajanda minejo 27](https://commons.wikimedia.org/wiki/File:%C4%88ajanda_minejo_27.jpg) | RG72 | CC BY-SA 4.0 |
| `t024.jpg` | [WLM - Peter J. Fontijn - Stoomgemaal De Tuut (28)](https://commons.wikimedia.org/wiki/File:WLM_-_Peter_J._Fontijn_-_Stoomgemaal_De_Tuut_(28).jpg) | Peter  Fontijn from NL | CC BY-SA 2.0 |
| `t031.jpg` | [2026-09-30 Велобайк-Бережковская, манометр 01](https://commons.wikimedia.org/wiki/File:2026-09-30_%D0%92%D0%B5%D0%BB%D0%BE%D0%B1%D0%B0%D0%B9%D0%BA-%D0%91%D0%B5%D1%80%D0%B5%D0%B6%D0%BA%D0%BE%D0%B2%D1%81%D0%BA%D0%B0%D1%8F,_%D0%BC%D0%B0%D0%BD%D0%BE%D0%BC%D0%B5%D1%82%D1%80_01.jpg) | Retired electrician | CC0 |
| `t032.jpg` | [Locomotora No. 106, Museo, Riotinto, Huelva, España, 2021 02](https://commons.wikimedia.org/wiki/File:Locomotora_No._106,_Museo,_Riotinto,_Huelva,_Espa%C3%B1a,_2021_02.jpg) | Benjamín Núñez González | CC BY-SA 4.0 |
| `t042.jpg` | [S-189 (submarine, 1954), Interior, San Petesburgo, Rusia, 2016 03](https://commons.wikimedia.org/wiki/File:S-189_(submarine,_1954),_Interior,_San_Petesburgo,_Rusia,_2016_03.jpg) | Benjamín Núñez González | CC BY-SA 4.0 |
| `t048.jpg` | [2017-10-05 06 56 51 Gas pressure indicator on the weather balloon infl](https://commons.wikimedia.org/wiki/File:2017-10-05_06_56_51_Gas_pressure_indicator_on_the_weather_balloon_inflation_nozzle_at_the_National_Weather_Service%27s_Baltimore-Washington_Weather_Forecast_Office_in_the_Dulles_section_of_Sterling,_Loudoun_County,_Virginia.jpg) | Famartin | CC BY-SA 4.0 |
| `t052.jpg` | [Manómetro de presión](https://commons.wikimedia.org/wiki/File:Man%C3%B3metro_de_presi%C3%B3n.jpg) | Frobles | CC BY-SA 4.0 |
| `t059.jpg` | [Painemittari1](https://commons.wikimedia.org/wiki/File:Painemittari1.jpg) | Kippari10 at fi.wikipedia | CC BY-SA 3.0 |
| `t064.jpg` | [2015-03-07 - Thinktank - Sasha Taylor - 288](https://commons.wikimedia.org/wiki/File:2015-03-07_-_Thinktank_-_Sasha_Taylor_-_288.JPG) | Sasha Taylor | CC BY-SA 4.0 |
| `t065.jpg` | [Budenberg Gauge LBS-PSI](https://commons.wikimedia.org/wiki/File:Budenberg_Gauge_LBS-PSI.JPG) | Dcolvin (talk) (Uploads) | CC0 |
| `t074.jpg` | [Manometre hpz](https://commons.wikimedia.org/wiki/File:Manometre_hpz.jpg) | Michel le tigre | CC BY-SA 3.0 |
| `t086.jpg` | [St. Marienkirchen-Polsenz Mostmuseum 1025](https://commons.wikimedia.org/wiki/File:St._Marienkirchen-Polsenz_Mostmuseum_1025.jpg) | Karl Gruber | CC BY-SA 4.0 |
| `t088.jpg` | [2015-03-07 - Thinktank - Sasha Taylor - 159](https://commons.wikimedia.org/wiki/File:2015-03-07_-_Thinktank_-_Sasha_Taylor_-_159.JPG) | Sasha Taylor | CC BY-SA 4.0 |
| `t094.jpg` | [Industrial instruments pressure gauge](https://commons.wikimedia.org/wiki/File:Industrial_instruments_pressure_gauge.jpg) | Palagiri | CC BY-SA 3.0 |
| `t100.jpg` | [Canadian Pacific pressure gauge (2800301396)](https://commons.wikimedia.org/wiki/File:Canadian_Pacific_pressure_gauge_(2800301396).jpg) | RAF-YYC from Calgary, Canada | CC BY-SA 2.0 |
| `t108.jpg` | [Schifffahrtstechnik im Verkehrsmuseum Dresden ..IMG 6713WI](https://commons.wikimedia.org/wiki/File:Schifffahrtstechnik_im_Verkehrsmuseum_Dresden_..IMG_6713WI.jpg) | Kora27 | CC BY-SA 4.0 |
| `t110.jpg` | [Pressure gauge on Siebe Gorman manual diver's pump P3220126](https://commons.wikimedia.org/wiki/File:Pressure_gauge_on_Siebe_Gorman_manual_diver%27s_pump_P3220126.jpg) | Peter Southwood | CC BY-SA 4.0 |
| `t062.jpg` | [Virginia V (ship, 1922) engine room 13 - Ashcroft Mfg. Co. vacuum gaug](https://commons.wikimedia.org/wiki/File:Virginia_V_(ship,_1922)_engine_room_13_-_Ashcroft_Mfg._Co._vacuum_gauge_(rotated).jpg) | Joe Mabel | CC BY-SA 4.0 |
| `t097.jpg` | [Locomotion No. 1, Beamish Museum, 11 September 2010 (2)](https://commons.wikimedia.org/wiki/File:Locomotion_No._1,_Beamish_Museum,_11_September_2010_(2).jpg) | dommylive from Peterlee, UK | CC BY 2.0 |

[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) and [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) texts apply to the files marked so. `eval/results/real_test_overlays.jpg` is a derivative of these photographs and carries the same attributions.
