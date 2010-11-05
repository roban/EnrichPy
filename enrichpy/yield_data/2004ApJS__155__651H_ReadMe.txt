J/ApJS/155/651      Evolution  of extremely metal-poor stars (Herwig+, 2004)
================================================================================
Evolution and yields of extremely metal-poor intermediate-mass stars.
    Herwig F.
   <Astrophys. J. Suppl. Ser., 155, 651-666 (2004)>
   =2004ApJS..155..651H
================================================================================
ADC_Keywords: Models, evolutionary ; Abundances
Keywords: nuclear reactions, nucleosynthesis, abundances -
          stars: AGB and post-AGB - stars: evolution - stars: interiors

Abstract:
    Intermediate-mass stellar evolution tracks from the main sequence to
    the tip of the AGB for five initial masses (2-6M_{sun}_) and
    metallicity Z=0.0001 have been computed. The detailed one-dimensional
    structure and evolution models include exponential overshooting, mass
    loss, and a detailed nucleosynthesis network with updated nuclear
    reaction rates. The network includes a two-particle heavy neutron sink
    for approximating neutron density in the He-shell flash. It is shown
    how the neutron-capture nucleosynthesis is important in models of very
    low metallicity for the formation of light neutron-heavy species, like
    sodium or the heavy neon and magnesium isotopes. The models have high
    resolution, as required for modeling the third dredge-up. All
    sequences have been followed from the pre-main sequence to the end of
    the AGB when all envelope mass is lost. Detailed structural and
    chemical model properties as well as yields are presented. This set 
    of stellar models is based on standard assumptions and updated input
    physics. It can be confronted with observations of extremely
    metal-poor stars and may be used to assess the role of AGB stars in
    the origin of abundance anomalies of some globular cluster members of
    correspondingly low metallicity.

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
table1.dat     81      101   Reaction network
table5.dat     63       23   Yields
table6.dat    299      504   Structure and abundance evolution of computed
                              tracks
--------------------------------------------------------------------------------

See also:
         VI/65  : Evolutionary models of evolved stars (Dorman+ 1993)
         VI/96  : Evolutionary Sequences (Fagotto+ 1993-96)
         VI/109 : Population Synthesis Models at very low metallicities
                                                              (Schaerer, 2003)
         VI/118 : Stellar Models until He burning (Claret+ 1995-1998)
 J/A+A/299/755  : Stellar evolution. II. Post-AGB (Bloecker+, 1995)
 J/A+A/401/1063 : Evolutionary synthesis models. III. (Anders+, 2003)
 J/A+A/424/919  : Stellar models grids. Z=0.02, M=0.8 to 125 (Claret, 2004)
 J/A+A/440/647  : New grids of stellar models. II. (Claret, 2005)
 J/A+A/453/769  : New grids of stellar models. III. (Claret+, 2006)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 37  A37   ---     Type      Type of reaction
  39- 41  I3    ---     ID        Identification number of the reaction
  43- 79  A37   ---     React     Reaction description
      81  A1    ---   r_React     Reference for React (1)
--------------------------------------------------------------------------------
Note (1): References as follows:
      1 = NACRE adopted;
      2 = Horiguchi et al. (1996, Chart of the Nuclides (Tokai: Japan
          Atomic Energy Research Inst.)
      3 = Caughlan & Fowler (1988, At. Data Nucl. Data Tables, 40, 283)
      4 = Iliadis et al. (2001ApJS..134..151I)
      5 = Bao et al. (2000, At. Data Nucl. Data Tables, 76, 70)
      6 = Beer et al. (2001, Nucl. Phys. A, 705, 239)
      7 = Bao et al. (2000, At. Data Nucl. Data Tables, 75, 1)
      8 = Hauser-Feshbach (Jorissen & Gorely, 2001, Nucl. Phys. A, 688, 508)
      9 = see Jorissen & Goriely (2001, Nucl. Phys. A, 688, 508)
      a = Wiescher et al. (1990ApJ...363..340W)
      b = Koehler & O'Brien (1989, Phys. Rev. C, 39, 1655)
      c = Takahashi Yokoi (1987, At. Data Nucl. Data Tables, 36, 375)
      d = Goriely (1999, Cat. <J/A+A/342/881>)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table5.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  2  I2    ---     Num       Sequential number of the species
   4-  8  A5    ---     Species   Species name (as in table6)
  10- 19  E10.4 solMass E82       The 2 solar mass (E82) model yield (1)
  21- 30  E10.4 solMass E84       The 3 solar mass (E84) model yield (1)
  32- 41  E10.4 solMass E85       The 4 solar mass (E85) model yield (1)
  43- 52  E10.4 solMass E79       The 5 solar mass (E79) model yield (1)
  54- 63  E10.4 solMass E86       The 6 solar mass (E86) model yield (1)
--------------------------------------------------------------------------------
Note (1): Normalized formed and ejected mass of element integrated
          over the stellar lifetime
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table6.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  3  A3    ---     Name      Sequence name (E79, E82, E84, E85, E86)
   5-  9  I5    ---     Model     Model number
  11- 16  F6.4  solMass Mass      Stellar mass
  18- 32  E15.8 yr      Age       Age (Zero at first TP (thermal-pulse))
  34- 39  F6.4  [K]     logTeff   Log of the effective temperature
  41- 46  F6.4 [solLum] logL      Log of the luminosity in solar units
  48- 57  E10.4 ---     H         H surface abundance
  59- 68  E10.4 ---     He4       He4 surface abundance
  70- 79  E10.4 ---     C12       C12 surface abundance
  81- 90  E10.4 ---     C13       C13 surface abundance
  92-101  E10.4 ---     N14       N14 surface abundance
 103-112  E10.4 ---     N15       N15 surface abundance
 114-123  E10.4 ---     O16       O16 surface abundance
 125-134  E10.4 ---     O17       O17 surface abundance
 136-145  E10.4 ---     O18       O18 surface abundance
 147-156  E10.4 ---     Ne20      Ne20 surface abundance
 158-167  E10.4 ---     Ne21      Ne21 surface abundance
 169-178  E10.4 ---     Ne22      Ne22 surface abundance
 180-189  E10.4 ---     Na23      Na23 surface abundance
 191-200  E10.4 ---     Mg24      Mg24 surface abundance
 202-211  E10.4 ---     Mg25      Mg25 surface abundance
 213-222  E10.4 ---     Mg26      Mg26 surface abundance
 224-233  E10.4 ---     Al26G     Al26G surface abundance
 235-244  E10.4 ---     Al27      Al27 surface abundance
 246-255  E10.4 ---     Si28      Si28 surface abundance
 257-266  E10.4 ---     Si29      Si29 surface abundance
 268-277  E10.4 ---     Si30      Si30 surface abundance
 279-288  E10.4 ---     G63       G63 surface abundance
 290-299  E10.4 ---     L1        L1 surface abundance
--------------------------------------------------------------------------------

History:
    From electronic version of the journal
================================================================================
(End)                  Greg Schwarz [AAS], Patricia Vannier [CDS]    26-Jan-2007
