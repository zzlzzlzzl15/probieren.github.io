### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3a13e3c1-8d75-46a3-93d3-5e22df452328
begin
	using PlutoUI, LaTeXStrings
	using WGLMakie
	using PlutoUI.ExperimentalLayout: grid, vbox, hbox, Div
	using StatsBase
	using MultivariateStats
	
	PlutoUI.TableOfContents(depth=2, indent=true, aside=true)
	using DSP, WAV, ImageIO
	using SampledSignals
end


# ╔═╡ f8c21b97-2cfc-4e4e-b838-46c447d6d9a5
md"""
# ICA： Independent Component Analysis

Author: Y. Yang, Z. Zhou, Y. Ma -- University Stuttgart. \
Supervisor: Prof. Benedikt Ehinger -- University Stuttagrt

"""

# ╔═╡ 7002562c-fd11-4da9-87ce-eeeb044f2cc5
html"""
 <style>
  div.plutoui-sidebar.aside {
   position: fixed;
   right: 1rem;
   top: 20rem;
   width: min(80vw, 23%);
   padding: 10px;
   border: 3px solid rgba(0, 0, 0, 0.15);
   border-radius: 10px;
   box-shadow: 0 0 11px 0px #00000010;
   max-height: calc(100vh - 5rem - 56px);
   overflow: auto;
   z-index: 40;
   background: transparent;
  }
 </style>

<style>
  main {
    max-width: 950px;
  }
  pluto-output img {
    width: 60%;
	height:60%;
  }

</style>
"""

# ╔═╡ 120b61e9-f3ec-416d-908f-312e16180190
begin
	s, fs = wavread("example1.wav")
	music_data1 = s[1:80000]
	s2, fs2 = wavread("taylor_enchanted_10sec.wav")
	music_data2 = s2[1:80000]
	selected_signalA_wave = @bind signalA_wave Select(["sin.(T1xs)","cos.(T1xs)"],default="sin.(T1xs)");
	selected_signalA_audio = @bind signalA_audio Select(["music_1"],default="music_1");
	selected_signalB_wave = @bind signalB_wave Select(["sin.(T2xs)","cos.(T2xs)"],default="cos.(T2xs)");
	selected_signalB_audio = @bind signalB_audio Select(["music_2"],default="music_2");
	signal_selection = @bind selection_function Radio(["1" => "wave", "2" => "audio"], default="1");
	
end;

# ╔═╡ 4823747d-0af6-47e1-aab5-643ff8b4125a
begin
	
	if selection_function=="1"
		
		selected_signalA = selected_signalA_wave
		selected_signalB = selected_signalB_wave
		signalA_box = md"""Choose the value for T1 $(@bind T1 PlutoUI.Slider(1:1:10;default=1,show_value=true))
				""";
		signalB_box = md"""Choose the value for T2 $(@bind T2 PlutoUI.Slider(1:1:10;default=1,show_value=true))
				""";
		signalA = signalA_wave
		signalB = signalB_wave
		
	elseif selection_function == "2"
		signalA_box = vbox([SampledSignals.SampleBuf(music_data1, fs)]);
		T1 = 0
		T2 = 0
		signalB_box = vbox([SampledSignals.SampleBuf(music_data2, fs)]);
		selected_signalA = selected_signalA_audio
		selected_signalB = selected_signalB_audio
		signalA = signalA_audio
		signalB = signalB_audio
	end
end;

# ╔═╡ f5198a60-52ab-4752-a959-d3b70a75022e
begin
	sidebar = Div([
		md"""
		##### Interactive Sliders
		Choose signal types
		""",
		hbox([signal_selection]),
		md"""
		---
		Select original signal A
		""",
		hbox([selected_signalA]),
		hbox([signalA_box]),
		
		md"""
		---
		Select original signal B
		""",
		hbox([selected_signalB]),
		hbox([signalB_box]),
		
		md"""
		---

		""",
		md"""Select mixing parameters a   $(@bind a PlutoUI.Slider(0:0.1:10;default=2,show_value=true))""",
		md"""Select mixing parameters b   $(@bind b PlutoUI.Slider(0:0.1:10;default=3.7,show_value=true))""",
		md"""Select mixing parameters c   $(@bind c PlutoUI.Slider(0:0.1:10;default=4.8,show_value=true))""",
		md"""Select mixing parameters d   $(@bind d PlutoUI.Slider(0:0.1:10;default=7.9,show_value=true))""",
		md"---",
		md"""mixed signal 1: aA+bB""",
		md"""mixed signal 2: cA+dB""",
	], 
		class = "plutoui-sidebar aside"
		
	);
end

# ╔═╡ 82c379cb-741d-4b37-b0b8-b9cea7e411e0
md"""
## 1. Why ICA: *The Cocktail-Party Problem*
"""

# ╔═╡ 2f878a3b-c55b-45a2-9a2b-fe88a99bd2c7
md"""
Imagine this situation: two microphones, held in different locations, are recording the speeches of two people speaking simultaneously. The microphones recorded two time signals, which could be denoted by $x_1(t)$ and $x_2(t)$, with $x_1$ and $x_2$ the amplitudes, and $t$ the time index. Each of these recorded signals is a weighted sum of the speech signals emitted by the two speakers, which we denote by $s_1(t)$ and $s_2(t)$. We could express this as a linear equation:

$$x_1(t)=a_{11} s_1 + a_{12} s_2$$
$$x_2(t)=a_{21} s_1 + a_{22} s_2$$

where $a_{11}$, $a_{12}$, $a_{21}$, and $a_{22}$ are parameters of the microphones. 

We can re-write the equations as a linear algebra form:

$$X = A S$$

It is very useful if $S$ can be estimated from $X$. 

Actually, if we knew the parameters $A$, we could solve the linear equation by classical methods: 

$$S = A^{-1} X$$

However, in most situations of EEG, $A$ is unknown, which makes the problem considerably more difficult. (the above is the so-called *Cocktail-Party Problem*)

ICA is often used to recover $S$ in this situation. 
"""

# ╔═╡ c01f44fa-2358-4455-9454-e3ee50118d3a
begin
	xxs = range(0, 10, length = 80000)

	
	if signalA == "sin.(T1xs)"
		A = sin.(T1*xxs)
		A_label = "sin"
	elseif signalA == "cos.(T1xs)"
		A = cos.(T1*xxs)
		A_label = "cos"
	elseif signalA =="music_1"
		A = music_data1
		A_label = "music_1"
	elseif signalA == "music_2"
		A = music_data2
		A_label = "music_2"	
	end;
	
	if signalB == "sin.(T2xs)"
		B = sin.(T2*xxs)
		B_label = "sin"
	elseif signalB == "cos.(T2xs)"
		B = cos.(T2*xxs)
		B_label = "cos"
	elseif signalB =="music_1"
		B = music_data2
		B_label = "music_1"
	elseif signalB == "music_2"
		B = music_data2
		B_label = "music_2"
	end;

	function plot_show(line_1, line_2, line1_label, line2_label, xs)

	f1 = Figure()
	g1a = f1[1,1]
	g1b = f1[2,2]
	g1 = f1[1, 2]
	lines(g1,xs, line_1, color = :skyblue)
	scatter!(g1, xs, line_1, color = :skyblue, label = line1_label, markersize = 4px)
	lines!(g1, xs ,line_2, color = :salmon)
	scatter!(g1, xs, line_2, color = :salmon, label = line2_label, markersize = 4px)
	axislegend()
	density(g1a, line_1, color = :skyblue)
	density(g1b, line_2, color = :salmon, direction = :y)
	g11 = f1[2, 1]
	scatter(g11, line_1, line_2, color = :green, markersize = 4, axis=(xlabel=L"f1(x)", ylabel=L"f2(x)"))
	f1


	end;
	
	function Identical_check(freq1, freq2, line1_label, line2_label)

		if line1_label == line2_label && freq1==freq2
			print("WARNING: ICA can't function well, because A and B are not independent.")
		end
		
		end;
		if A_label == "sin"||A_label == "cos"
			signal1_label = "$A_label($T1 xs）"
		elseif A_label == "music_1"||A_label == "music_2"
			signal1_label = A_label
		end
	
		if B_label == "sin"||B_label == "cos"
			signal2_label = "$B_label($T2 xs）"
		elseif B_label == "music_1"||B_label == "music_2"
			signal2_label = B_label
		end;
end;

# ╔═╡ 83cf2fd4-b1ca-40eb-b22e-14ba0f61208b
md"""
## 2. Examples
"""

# ╔═╡ dfc5ef72-0c3e-4099-b7b8-f594f5414df1
begin
	plot_show(A, B, signal1_label, signal2_label, xxs)
end

# ╔═╡ 24f27037-3c99-45ec-b22e-d2f4cfdce411
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ a406737f-e4f3-4cea-8cef-04b8f28d258b
begin
	if signalA!="music_1"||signalB!="music_2"||signalA!="music_1"||signalB!="music_2"
		Identical_check(T1, T2,A_label, B_label)
	end
end

# ╔═╡ 9dc65c9b-d949-46ad-b97e-d1a5f9ca0282
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ 7002562c-fd11-4da9-87ce-eeeb044f2cc5
html"""
 <style>
  div.plutoui-sidebar.aside {
   position: fixed;
   right: 1rem;
   top: 20rem;
   width: min(80vw, 23%);
   padding: 10px;
   border: 3px solid rgba(0, 0, 0, 0.15);
   border-radius: 10px;
   box-shadow: 0 0 11px 0px #00000010;
   max-height: calc(100vh - 5rem - 56px);
   overflow: auto;
   z-index: 40;
   background: transparent;
  }
 </style>

<style>
  main {
    max-width: 950px;
  }
  pluto-output img {
    width: 60%;
	height:60%;
  }

</style>
"""

# ╔═╡ eed4fc32-86d1-4bfd-a518-46617a79005f
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ d9c61e9e-7838-426b-9cfc-48f217ad692a
begin 
	md"""Select rotation angle $(@bind angle PlutoUI.Slider(0:π/6:2*π;default=π/2,show_value=true))"""
end

# ╔═╡ ee8c132b-d433-4be6-a7cb-f83a4a24e97f
md"""
### 2.4. Signal rotation

In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean space. For example, using the convention below, the matrix

$$A = \left[
		\begin{matrix}
			 cos\theta  \quad-sin\theta\\
			sin\theta \quad cos\theta
	  	\end{matrix}
	\right]
	  
		\left[
		\begin{matrix}
			x \\
			y 
	  	\end{matrix}
		\right]$$

the rotation angle will be \theta, from $$0^{。}$$ to $$180^{。}$$

"""

# ╔═╡ 09547104-2d92-4365-b4be-ba7729f2d65c
md"""
## 6. Sources

[HTML: What is ICA?](https://benediktehinger.de/ica/)

[PDF: Tutorial on ICA - Dominic Langlois, Sylvain Chartier, and Dominique Gosselin](https://www.tqmp.org/RegularArticles/vol06-1/p031/p031.pdf)

[HTML: ICA for dummies - Arnould Delorme](https://arnauddelorme.com/ica_for_dummies/)

[HTML: (fast)ICA Tutorial - Aapo Hyvärinen](https://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/)
"""

# ╔═╡ 24f27037-3c99-45ec-b22e-d2f4cfdce411
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ 9dc65c9b-d949-46ad-b97e-d1a5f9ca0282
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ 7d77dcf3-2a6c-4f9e-bfe4-1beeed7d66c3
md"""
## 5. Assumptions and Properties of ICA

### 5.1. Main Assumptions of ICA

* 1. Each source $s_m$ is statistically independent of the others;
* 2. The mixing matrix $A$ is square and of full rank, i.e. there must be the same number of signals to sources and they must be linearly independent of each other;
* 3. There is no external noise, i.e. the model is noise-free;
* 4. The data is centered;
* 5. There is maximally one gaussian source.

### 5.2. Properties of ICA

* 1. ICA can only separate linearly mixed sources;
* 2. Changing the order in which the points are plotted or changing the order of channel will have almost no effect on the outcome;
* 3. There is no external noise, i.e. the model is noise-free;
* 4. perfect Gaussian sources can not be separated;
* 5. Even when the sources are not independent, ICA finds a space where they are maximally independent.

"""

# ╔═╡ f3792960-284c-4df4-a60d-3da9ee9fb5c2
md"""
### 2.1. Audio

Sounds are waves and can be easily recorded as digital signals. When two sources are played simultaneously, what we recorded is one mixed-signal $x$. ICA can be used to disentangle the two recorded mixture $x$ into their original sources $s_1$ and $s_2$. 

### 2.2. EEG

In EEG, many different synchronous active neuronal patches (the sources) are assumed. And electrodes are always recorded as the summation of the sources. ICA can be used to separate the summation into original sources. 

### 2.3. Arbitrary Case

A more abstract case is to have two unique signals $s_1$ and $s_2$ and generate their mixture $x$ by weight-adding:

$$x_1 = s_1 - 2 s_2$$
$$x_2 = 1.73 s_1 + 3.41 s_2$$

To describe the above in a linear algebra matrix: $X = AS$, where:

$$X = [x_1, x_2]^{\top}$$
$$S = [s_1, s_2]^{\top}$$
$$A = \left[
		\begin{matrix}
			1 \quad -2 \\ 
			1.73 \quad 3.41
	  	\end{matrix}
	  \right]$$

"""

# ╔═╡ 3bc623f5-45a1-433c-b57d-f99503a9ef4f
begin
	line1 = a*A + b*B
	line2 = c*A + d*B
	
	
	plot_show(line1, line2, "$a $signal1_label+ $b $signal2_label", "$c $signal1_label+ $d $signal2_label", xxs)
end

# ╔═╡ 04bebdc9-73b5-4b90-91e7-f8121e4d2d29
if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(line1, fs)
end

# ╔═╡ 5c16823e-17ff-4a26-b403-f2b265b7a186
if selection_function == "2"
		#print("""mixed signal 2""")
		SampledSignals.SampleBuf(line2, fs)
<<<<<<< HEAD
end

# ╔═╡ ee8c132b-d433-4be6-a7cb-f83a4a24e97f
md"""
### 2.4. Signal rotation

In linear algebra, a rotation matrix is a transformation matrix that is used to perform a rotation in Euclidean space. For example, using the convention below, the matrix

$$A = \left[
		\begin{matrix}
			 cos\theta  \quad-sin\theta\\
			sin\theta \quad cos\theta
	  	\end{matrix}
	\right]
	  
		\left[
		\begin{matrix}
			x \\
			y 
	  	\end{matrix}
		\right]$$

the rotation angle will be \theta, from $$0^{。}$$ to $$180^{。}$$

"""

# ╔═╡ d9c61e9e-7838-426b-9cfc-48f217ad692a
begin 
	md"""Select rotation angle $(@bind angle PlutoUI.Slider(0:10:360;default=π/2,show_value=true)) """
=======
	end
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709
end

# ╔═╡ 7b5925b5-5c81-4faf-b224-d9f5f828a4a3
begin
<<<<<<< HEAD
	angle_pi = angle/180*pi
	original_signal = cat(line1, line2, dims=2)
	rotation_matrix = [cos.(angle_pi) -sin.(angle_pi);sin.(angle_pi) cos.(angle_pi)]
	rotated_signal = rotation_matrix*transpose(original_signal)
end;
=======
	original_signal = cat(line1, line2, dims=2)
	rotation_matrix = [cos.(angle) -sin.(angle);sin.(angle) cos.(angle)]
	rotated_signal = rotation_matrix*transpose(original_signal)
end

# ╔═╡ 886e6bf3-5d76-4a3f-a13a-48141af3562a
begin
	if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(rotated_signal[1,:], fs)
	end
end
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

# ╔═╡ 03df5cdb-a9f9-4931-913b-1b666be141bf
begin
	plot_show(rotated_signal[1,:], rotated_signal[2,:], "Rotated signal 1", "Rotated signal 2", xxs)
end

<<<<<<< HEAD
# ╔═╡ e5fabaa0-b6bd-4dfc-b8e2-11ce5d3030c9
md"""
## 3. ICA Preprocesses
"""
=======
# ╔═╡ a31bef66-e441-4389-b4db-a86ba6d01c79
begin
	mean1 = mean(line1)
	mean2 = mean(line2)
	
	line1_demean = line1 - fill(mean1, (80000,1))
	line2_demean = line2 - fill(mean2, (80000,1)) #change

	demeaning_matrix = transpose(cat(line1_demean, line2_demean, dims=2))
	M = fit(Whitening, demeaning_matrix)
	
	whitening_matrix = transpose(M.W) * demeaning_matrix
	
	plot_show(whitening_matrix[1,:], whitening_matrix[2,:], "whiten Mixed signal 1", "whiten Mixed signal 2")
end

# ╔═╡ 3e4639ca-2a42-48af-955a-973d1e3d8483
begin
	unmixing_W = fit(ICA, demeaning_matrix, 2, do_whiten = true)
	unmixing_matrix = unmixing_W.W
	unmixed_data = transpose(unmixing_matrix) * demeaning_matrix
	
	unmixed_line1 = unmixed_data[1,:]
	unmixed_line2 = unmixed_data[2,:]
	
	f4 = Figure()
	g4 = f4[2, 1]
	g4a = f4[1,1]
	g4b = f4[2,2]
	
	density(g4a, unmixed_line1, color = :skyblue)
	density(g4b, unmixed_line2, color = :salmon, direction = :y)
	lines(g4, xs, unmixed_line1, color = :skyblue, label = "Recovered signal 1")
	lines!(g4, xs, unmixed_line2, color = :salmon, label = "Recovered signal 2")
	axislegend();
	
	g44 = f4[1, 2]
	scatter(g44, unmixed_line1, unmixed_line2, color = :green, markersize = 4, axis=(; xlabel=L"f1(x)", ylabel=L"f2(x)"))
	f4

	plot_show(unmixed_line1, unmixed_line2, "Recovered signal 1", "Recovered signal 2")
end

# ╔═╡ b6acb9e8-5f9d-47af-948b-244c3c90ab42
begin
	if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(unmixed_line2, fs)
	end
end

# ╔═╡ 0bc39b2a-397e-48d5-bb6c-f489334f666c
begin
	if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(unmixed_line1, fs)
	end
end

# ╔═╡ 480d8366-40f3-4f35-b19e-8dd456b2e88b
whitening_matrix[2,:];
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

# ╔═╡ 17f4bfe0-8a50-490b-a85b-e465bbd707e6
md"""

### 3.1. Preprocess Step 1. Demeaning

Signals always have arbitrary mean values. Normally the zero-mean value is wanted for further calculation. 

"Demeaning" the signals means to shift the baseline to $0$ for each channel:

$$x_{m}=x-E[x]$$

### 3.2. Preprocess Step 2. Whitening

Almost every mixed signal has been stretched and rotated. ICA can "un-rotate" the signal, but must get the "un-stretched" (also called "whitened" or "sphered") signal first. 

After "Preprocess Step 1.", the mixed signal is now zero-mean-valued but $s_1$ and $s_2$ are still correlated. Whitening forces them to be uncorrelated by changing the coordinate of the mixed data $X$ linearly.

#### \* The Math Behind Whitening
A whitened signal means its covariance is the identity matrix $I$. 

To whiten the signal means to find a transformation $V$, so that for $x_m$, its transformation $x_w = V x_m$ has covariance of $I$. 

Covariance of a matrix $N$ is defined as: $Cov(N) = \frac{1}{n-1}(N N^{\top})$

Donate: $C = Cov(x_m)$, and $V = C^{-\frac{1}{2}}$, and calculate: 

$$\begin{split}
Cov(x_w) &= Cov(V x_m) \\
&= Cov(C^{-\frac{1}{2}} x_m) \\
&= (C^{-\frac{1}{2}} x_m)(C^{-\frac{1}{2}} x_m)^{\top} \\
&= C^{-\frac{1}{2}} x_m x_m^{\top} C^{-\frac{1}{2}} \\
&= C^{-\frac{1}{2}} C C^{-\frac{1}{2}} \\
&= I
\end{split}$$

Thus $x_w = V x_m = C^{-\frac{1}{2}} x_m$ is the whitened signal.
"""

# ╔═╡ add81a72-7074-4ba7-981b-e45644d3d980
begin
<<<<<<< HEAD
	mean1 = mean(line1)
	mean2 = mean(line2)
	
	line1_demean = line1 - fill(mean1, (80000,1))
	line2_demean = line2 - fill(mean2, (80000,1)) #change

	demeaning_matrix = transpose(cat(line1_demean, line2_demean, dims=2))
	M = fit(Whitening, demeaning_matrix)
	
	whitening_matrix = transpose(M.W) * demeaning_matrix
	
	plot_show(whitening_matrix[1,:], whitening_matrix[2,:], "whiten Mixed signal 1", "whiten Mixed signal 2", xxs)
end

# ╔═╡ 480d8366-40f3-4f35-b19e-8dd456b2e88b
whitening_matrix[2,:];

# ╔═╡ a6c7d526-3f6d-47d2-b3ec-fc9c56d67ca5
whitening_matrix[1,:];

=======
	if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(rotated_signal[2,:], fs)
	end
end

>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709
# ╔═╡ 26f37ed7-ac54-4dc2-9e7a-b09afd4a51c9
md"""
## 4. ICA Algorithms

ICA tries to restore the original signals from their mixture by "Rotating" it. There are typically two approaches to getting the right rotation: 1. Maximizing statistical independence and 2. Minimizing normality.


### 4.1.1. Maximizing Statistical Independence (for two signals)

Statistical independence can be measured by mutual information. It tells how much information one can get about an outcome of B if A is already known. When minimizing mutual information, statistical independence is maximized.

#### \* Mutual Information
Mutual information $I(X; Y)$ measures the uncertainty of two variables $X$ and $Y$: 

$$\begin{split}
I(X; Y) &= H (X) - H(X|Y) \\
&= H (X) - (H(X, Y) - H(Y)) \\
&= -\sum_x P(x) \log P(x) - (-\sum_x P(x, y) \log P(x, y) - (-\sum_x P(y) \log P(y))\\
\end{split}$$

Mutual information $I(X; Y) = 0$ means $X$ and $Y$ are independent.

### 4.1.2. InfoMax Algorithm

Using the definition of mutual information, InfoMax Algorithm was proposed by Amari et
al. (1996) to calculate the unmixing matrix $W$:

* 1. Initialize $W(0)$ randomly.
* 2. Compute $W(t+1) = W(t) + \eta(t) (I - f(Y) Y^{\top}) W(t)$.
* 3. If not converged, go back to step 2.

where $t$ represents the sampling step, $\eta(t)$ a general function that specifies the size of the steps for the unmixing matrix updates (usually an exponential function or a constant), $f(Y)$ a nonlinear function usually chosen according to the type of distribution (super or sub-Gaussian). 

### 4.2.1. Minimizing Normality (for multiple signals)

The central limit theorem (CLT) states that the mixture (signals) of two independent distributions (sources) often tends to be more normal-distributed than the underlying signals. Therefore if the "normal-distributedness" is minimized, the independence of the two sources may appear. 

Normal-distributiveness is also called gaussianity or normality, which can be measured by a non-robust method called "kurtosis". The more kurtosis close to $0$, the more normality the mixture signal is. Thus, to minimize the normality of the mixture, a large value of kurtosis is preferred， which leads to independent restored signals. 


#### \* Definition of Kurtosis

Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution: 

$$Kurt[X] = \frac{\sum^N_{i=1} (X_i - E[X])^4 / N}{s^4} - 3$$

where $E[X]$ is the mean of $X$, $s$ is the standard deviation, and $N$ is the number of data points.

### 4.2.2. Fast ICA Algorithm

* 1. Initialize $w_i$ randomly.
* 2. Calculate a temporary variable: $w_i^+ = E(\phi'(w_i^{\top} X))w_i - E(x \phi(w_i^{\top} X))$, where $\phi(\cdot)$ is a non-quadratic function (usually $\tanh(\cdot)$), and $\phi'(\cdot)$ the derivative of $\phi(\cdot)$.
* 3. Calculate $w_i$ using $w_i^+$: $w_i=\frac{w_i^+}{||w_i^+||}$.
* 4. For $i = 1$, go to **step 7.**. Else continue with **step 5.**.
* 5. Update the temporary variable: $w_i^+ = w_i - \sum_{j=1}^{i-1} w_i^{\top} w_j w_i$.
* 6. Calculate $w_i$ again: $w_i=\frac{w_i^+}{||w_i^+||}$.
* 7. If not converged, go back to **step 2.**. Else go back to **step 1.** with $i = i + 1$ until all components are extracted.


Once a given $w_i$ has converged, the next one ($w_{i+1}$) must be made orthogonal to all others previously extracted.
"""

<<<<<<< HEAD
# ╔═╡ 3e4639ca-2a42-48af-955a-973d1e3d8483
begin
	unmixing_W = fit(ICA, demeaning_matrix, 2, do_whiten = true)
	unmixing_matrix = unmixing_W.W
	unmixed_data = transpose(unmixing_matrix) * demeaning_matrix
	
	unmixed_line1 = unmixed_data[1,:]
	unmixed_line2 = unmixed_data[2,:]
	
	f4 = Figure()
	g4 = f4[2, 1]
	g4a = f4[1,1]
	g4b = f4[2,2]
	
	density(g4a, unmixed_line1, color = :skyblue)
	density(g4b, unmixed_line2, color = :salmon, direction = :y)
	lines(g4, xxs, unmixed_line1, color = :skyblue, label = "Recovered signal 1")
	lines!(g4, xxs, unmixed_line2, color = :salmon, label = "Recovered signal 2")
	axislegend();
	
	g44 = f4[1, 2]
	scatter(g44, unmixed_line1, unmixed_line2, color = :green, markersize = 4, axis=(; xlabel=L"f1(x)", ylabel=L"f2(x)"))
	f4

	plot_show(unmixed_line1, unmixed_line2, "Recovered signal 1", "Recovered signal 2", xxs)
end

# ╔═╡ 0bc39b2a-397e-48d5-bb6c-f489334f666c
begin
	if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(unmixed_line1, fs)
	end
end;

# ╔═╡ 2b8e5c7f-2d69-464e-90b4-6985ac49a17d
if selection_function == "2"
		#print("""mixed signal 1""")
		SampledSignals.SampleBuf(rotated_signal[1,:], fs)
end

# ╔═╡ add81a72-7074-4ba7-981b-e45644d3d980
begin
	if selection_function == "2"
		#print("""mixed signal 2""")
		SampledSignals.SampleBuf(rotated_signal[2,:], fs)
	end
end

# ╔═╡ 0dd2a96b-7296-4870-adca-1cfac8791216
Identical_check(T1, T2, A_label, B_label)

# ╔═╡ 7d77dcf3-2a6c-4f9e-bfe4-1beeed7d66c3
=======
# ╔═╡ e5fabaa0-b6bd-4dfc-b8e2-11ce5d3030c9
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709
md"""
## 3. ICA Preprocesses
"""

<<<<<<< HEAD
# ╔═╡ 09547104-2d92-4365-b4be-ba7729f2d65c
md"""
## 6. Sources

[HTML: What is ICA?](https://benediktehinger.de/ica/)

[PDF: Tutorial on ICA - Dominic Langlois, Sylvain Chartier, and Dominique Gosselin](https://www.tqmp.org/RegularArticles/vol06-1/p031/p031.pdf)

[HTML: ICA for dummies - Arnould Delorme](https://arnauddelorme.com/ica_for_dummies/)

[HTML: (fast)ICA Tutorial - Aapo Hyvärinen](https://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/)
"""
=======
# ╔═╡ a6c7d526-3f6d-47d2-b3ec-fc9c56d67ca5
whitening_matrix[1,:];
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SampledSignals = "bd7594eb-a658-542f-9e75-4c4d8908c167"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
WAV = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
DSP = "~0.7.6"
ImageIO = "~0.6.6"
LaTeXStrings = "~1.3.0"
MultivariateStats = "~0.9.1"
PlutoUI = "~0.7.39"
SampledSignals = "~2.1.3"
StatsBase = "~0.33.20"
WAV = "~1.2.0"
WGLMakie = "~0.6.12"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "5c0b629df8a5566a06f5fef5100b53ea56e465a0"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.2"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
<<<<<<< HEAD
git-tree-sha1 = "0582b5976fc76523f77056e888e454f0f7732596"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.22"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"
=======
git-tree-sha1 = "621913bff3923ff489e4268ba2b425bfacbb1759"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.21"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "8d9e48436c5589fbd51ae8c8165a299a219188c0"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.15"
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["Printf", "ScanByte", "TranscodingStreams"]
git-tree-sha1 = "d50976f217489ce799e366d9561d56a98a30d7fe"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "0.8.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "3fb5d9183b38fdee997151f723da42fb83d1c6f2"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.6"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "aafa0665e3db0d3d0890cdc8191ea03dc279b042"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.66"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "03b753748fd193a7f2730c02d880da27c5a24508"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "94f5101b96d2d968ace56f7f2db19d0a5f592e28"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.15.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "b5c7fe9cea653443736d264b85466bad8c574f4a"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.9"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78e2c69783c9753a91cdae88a8d432be85a2ab5e"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "53c7e69a6ffeb26bd594f5a1421b889e7219eeaa"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.9.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f025b79883f361fa1bd80ad132773161d231fd9f"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "23e651bbb8d00e9971015d0dd306b780edbdb6b9"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.3"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "bcf640979ee55b652f3b01650444eb7bbe3ea837"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "fd6f0cae36f42525567108a42c1c674af2ac620d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.5"

[[deps.JSServe]]
deps = ["Base64", "CodecZlib", "Colors", "HTTP", "Hyperscript", "JSON3", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "Test", "UUIDs", "WebSockets", "WidgetsBase"]
git-tree-sha1 = "961c49293ac6b4e44df3b73bca89c76913ef6b4a"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f9"
version = "1.2.7"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
<<<<<<< HEAD
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"
=======
git-tree-sha1 = "0a7ca818440ce8c70ebb5d42ac4ebf3205675f04"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.4"
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
<<<<<<< HEAD
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"
=======
git-tree-sha1 = "7c88f63f9f0eb5929f15695af9a4d7d3ed278a91"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.16"
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.Makie]]
deps = ["Animations", "Base64", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MakieCore", "Markdown", "Match", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "RelocatableFolders", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "c27ed640732b1e9bd7bb8f40d987873d8f5b4bca"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.17.12"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "9bd42b962d5c6182fa0d74b1970edb075fe313e5"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.3.6"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Match]]
git-tree-sha1 = "1d9bc5c1a6e7ee24effb93f175c9342f9154d97f"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.2.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "Test"]
git-tree-sha1 = "114ef48a73aea632b8aebcb84f796afcc510ac7c"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.4.3"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
<<<<<<< HEAD
git-tree-sha1 = "14cb991ee7ccc6dabda93d310400575c3cae435b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.2"
=======
git-tree-sha1 = "9f4f5a42de3300439cb8300236925670f844a555"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.1"
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "a8cbf066b54d793b9a48c5daa5d586cf2b5bd43d"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.1.0"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "7008a3412d823e29d370ddc77411d593bd8a3d03"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e925a64b8585aa9f4e3047b8d2cdc3f0e79fd4e4"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.16"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "1155f6f937fa2b94104162f01fa400e192e4272f"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.2"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
<<<<<<< HEAD
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3010a6dd6ad4c7384d2f38c58fa8172797d879c1"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.0"
=======
deps = ["LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "d6de04fd2559ecab7e9a683c59dcbc7dbd20581a"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.1.5"
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "307761d71804208c0c62abdbd0ea6822aa5bbefd"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.2.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMD]]
git-tree-sha1 = "7dbc15af7ed5f751a82bf3ed37757adf76c32402"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.1"

[[deps.SampledSignals]]
deps = ["Base64", "Compat", "DSP", "FFTW", "FixedPointNumbers", "IntervalSets", "LinearAlgebra", "Random", "TreeViews", "Unitful"]
git-tree-sha1 = "df45b2fbce4377e66c520df2779f7f7c9ca64291"
uuid = "bd7594eb-a658-542f-9e75-4c4d8908c167"
version = "2.1.3"

[[deps.ScanByte]]
deps = ["Libdl", "SIMD"]
git-tree-sha1 = "2436b15f376005e8790e318329560dcc67188e84"
uuid = "7b38b023-a4d7-4c5e-8d43-3f3097f304eb"
version = "0.3.3"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "6b5bba824b515ec026064d1e7f5d61432e954b71"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.2.9"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f94f9d627ba3f91e41a815b9f9f977d729e2e06f"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "23368a3313d12a2326ad0035f0db0c0966f438ef"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "66fe9eb253f910fe8cf161953880cfdaef01cdf0"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.0.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8d7530a38dbd2c397be7ddd01a424e4f411dcc41"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.2"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "0005d75f43ff23688914536c5e9d5ac94f8077f7"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.20"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "ec47fb6069c57f1cee2f67541bf8f23415146de7"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.11"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "d24a825a95a6d98c385001212dc9020d609f2d4f"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.8.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "fcf41697256f2b759de9380a7e8196d6516f0310"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.WAV]]
deps = ["Base64", "FileIO", "Libdl", "Logging"]
git-tree-sha1 = "7e7e1b4686995aaf4ecaaf52f6cd824fa6bd6aa5"
uuid = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
version = "1.2.0"

[[deps.WGLMakie]]
deps = ["Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "ImageMagick", "JSServe", "LinearAlgebra", "Makie", "Observables", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "4c6a0919bf4cf07209ec505605f8594eb9372873"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.6.12"

[[deps.WebSockets]]
deps = ["Base64", "Dates", "HTTP", "Logging", "Sockets"]
git-tree-sha1 = "f91a602e25fe6b89afc93cf02a4ae18ee9384ce3"
uuid = "104b5d7c-a370-577a-8038-80a2059c5097"
version = "1.5.9"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "78736dab31ae7a53540a6b752efc61f77b304c5b"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.8.6+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─3a13e3c1-8d75-46a3-93d3-5e22df452328
# ╟─f8c21b97-2cfc-4e4e-b838-46c447d6d9a5
# ╟─7002562c-fd11-4da9-87ce-eeeb044f2cc5
# ╠═120b61e9-f3ec-416d-908f-312e16180190
# ╠═4823747d-0af6-47e1-aab5-643ff8b4125a
<<<<<<< HEAD
# ╠═f5198a60-52ab-4752-a959-d3b70a75022e
# ╟─82c379cb-741d-4b37-b0b8-b9cea7e411e0
# ╟─2f878a3b-c55b-45a2-9a2b-fe88a99bd2c7
# ╟─c01f44fa-2358-4455-9454-e3ee50118d3a
# ╟─83cf2fd4-b1ca-40eb-b22e-14ba0f61208b
# ╟─dfc5ef72-0c3e-4099-b7b8-f594f5414df1
# ╟─04bebdc9-73b5-4b90-91e7-f8121e4d2d29
# ╟─5c16823e-17ff-4a26-b403-f2b265b7a186
# ╟─24f27037-3c99-45ec-b22e-d2f4cfdce411
# ╟─0bc39b2a-397e-48d5-bb6c-f489334f666c
# ╟─a406737f-e4f3-4cea-8cef-04b8f28d258b
# ╟─9dc65c9b-d949-46ad-b97e-d1a5f9ca0282
# ╟─f3792960-284c-4df4-a60d-3da9ee9fb5c2
# ╟─3bc623f5-45a1-433c-b57d-f99503a9ef4f
# ╟─ee8c132b-d433-4be6-a7cb-f83a4a24e97f
# ╟─d9c61e9e-7838-426b-9cfc-48f217ad692a
# ╟─7b5925b5-5c81-4faf-b224-d9f5f828a4a3
# ╟─03df5cdb-a9f9-4931-913b-1b666be141bf
# ╟─e5fabaa0-b6bd-4dfc-b8e2-11ce5d3030c9
# ╟─17f4bfe0-8a50-490b-a85b-e465bbd707e6
# ╟─480d8366-40f3-4f35-b19e-8dd456b2e88b
# ╟─a31bef66-e441-4389-b4db-a86ba6d01c79
# ╟─a6c7d526-3f6d-47d2-b3ec-fc9c56d67ca5
# ╟─26f37ed7-ac54-4dc2-9e7a-b09afd4a51c9
# ╟─3e4639ca-2a42-48af-955a-973d1e3d8483
# ╟─2b8e5c7f-2d69-464e-90b4-6985ac49a17d
# ╟─add81a72-7074-4ba7-981b-e45644d3d980
# ╟─0dd2a96b-7296-4870-adca-1cfac8791216
# ╟─7d77dcf3-2a6c-4f9e-bfe4-1beeed7d66c3
# ╠═09547104-2d92-4365-b4be-ba7729f2d65c
=======
# ╠═82c379cb-741d-4b37-b0b8-b9cea7e411e0
# ╠═2f878a3b-c55b-45a2-9a2b-fe88a99bd2c7
# ╠═809cf99b-9dfb-41b0-9dac-03193486ac45
# ╠═adb62abf-6bcf-425a-9121-4fac936a6757
# ╠═1a93e5f6-aa0c-4e36-88a9-77f01c3f9378
# ╠═923cc527-b59b-4e09-8d87-e75042e81605
# ╠═7b311f12-1c75-41a0-b7e1-169df0b72421
# ╠═8d7e08d1-a90f-48fa-9d4d-1a53a704aaf2
# ╠═dfc5ef72-0c3e-4099-b7b8-f594f5414df1
# ╟─a406737f-e4f3-4cea-8cef-04b8f28d258b
# ╟─83cf2fd4-b1ca-40eb-b22e-14ba0f61208b
# ╠═7002562c-fd11-4da9-87ce-eeeb044f2cc5
# ╠═eed4fc32-86d1-4bfd-a518-46617a79005f
# ╠═886e6bf3-5d76-4a3f-a13a-48141af3562a
# ╠═90151eed-b00f-4b25-ab1c-aa226b84d09a
# ╠═d9c61e9e-7838-426b-9cfc-48f217ad692a
# ╠═b6acb9e8-5f9d-47af-948b-244c3c90ab42
# ╠═3e4639ca-2a42-48af-955a-973d1e3d8483
# ╠═480d8366-40f3-4f35-b19e-8dd456b2e88b
# ╠═03df5cdb-a9f9-4931-913b-1b666be141bf
# ╠═ee8c132b-d433-4be6-a7cb-f83a4a24e97f
# ╠═09547104-2d92-4365-b4be-ba7729f2d65c
# ╠═01ad2a40-8907-4b05-90be-e15151efbdb4
# ╠═24f27037-3c99-45ec-b22e-d2f4cfdce411
# ╠═9dc65c9b-d949-46ad-b97e-d1a5f9ca0282
# ╠═7b5925b5-5c81-4faf-b224-d9f5f828a4a3
# ╠═7d77dcf3-2a6c-4f9e-bfe4-1beeed7d66c3
# ╠═a31bef66-e441-4389-b4db-a86ba6d01c79
# ╠═f3792960-284c-4df4-a60d-3da9ee9fb5c2
# ╠═0bc39b2a-397e-48d5-bb6c-f489334f666c
# ╠═3bc623f5-45a1-433c-b57d-f99503a9ef4f
# ╠═17f4bfe0-8a50-490b-a85b-e465bbd707e6
# ╠═add81a72-7074-4ba7-981b-e45644d3d980
# ╠═26f37ed7-ac54-4dc2-9e7a-b09afd4a51c9
# ╠═e5fabaa0-b6bd-4dfc-b8e2-11ce5d3030c9
# ╠═a6c7d526-3f6d-47d2-b3ec-fc9c56d67ca5
>>>>>>> c5fabc8de963e2436a37bf81bf89009c1a90d709
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
