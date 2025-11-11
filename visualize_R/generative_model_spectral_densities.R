library(jsonlite)
library(R.utils)
library(vscDebugger)
library(ggplot2)
library(ggridges)
library(hrbrthemes)
library(dplyr)
library(tidyr)
library(viridis)

json_gz_file_path <- file.path("/home/giannis/Documents/data_for_figures/sim_vowels_figures.json.gz") # nolint: line_length_linter.
json_file_path <- gsub(".gz$", "", json_gz_file_path)

if (!file.exists(json_file_path)) {
  # Decompress the .json.gz file
  json_file_path <- gunzip(json_gz_file_path, remove = FALSE)
}

# Load the JSON data
data <- fromJSON(json_file_path)

keys <- names(data)
subkeys <- names(data[[keys[1]]])

# Plot all vowels for a random speaker
a_spectral_density <- data[["a_0.705"]][["spectral_density"]]
a_spectral_density <- a_spectral_density[1:floor(length(a_spectral_density)/2+1)] # nolint
e_spectral_density <- data[["e_0.705"]][["spectral_density"]][1:length(a_spectral_density)]
i_spectral_density <- data[["I_0.705"]][["spectral_density"]][1:length(a_spectral_density)]
aw_spectral_density <- data[["aw_0.705"]][["spectral_density"]][1:length(a_spectral_density)]
u_spectral_density <- data[["u_0.705"]][["spectral_density"]][1:length(a_spectral_density)]
freq_axis <- data[["a_0.705"]][["freq_axis"]][1:length(a_spectral_density)]

data_densities <- data.frame()
for (vowel_key in grep("_0\\.", names(data), value = TRUE)) {
  vowel_type <- sub("_.*", "", vowel_key)  # Extract vowel type (a, e, i, etc.)
  dens <- data[[vowel_key]][["spectral_density"]][1:length(a_spectral_density)]
  freq <- data[[vowel_key]][["freq_axis"]][1:length(a_spectral_density)]
  df_temp <- data.frame(
    freq = freq,
    density = dens,
    vowel = vowel_type
  )
  data_densities <- rbind(data_densities, df_temp)
}

data_sample <- data.frame(
  freq = freq_axis,
  a = a_spectral_density,
  e = e_spectral_density,
  i = i_spectral_density,
  aw = aw_spectral_density,
  u = u_spectral_density
)

data_long <- data_sample %>%
  gather(key = "vowel", value = "spectral_density", - freq)

data_sample <- data_sample %>%
  gather(key = "text", value = "value") %>%
  mutate(text = gsub("\\.", " ", text)) %>%
  mutate(value = as.numeric(value))

# Simple linear plots

p1 <- ggplot(data = data_long, aes(x = freq, y = spectral_density, color = vowel, fill = vowel)) + # nolint
  geom_area(alpha = 0.6) +
  geom_line() +
  theme_ipsum() +
  labs(
    x = "Frequency (Hz)",
    y = "Spectral Density",
    title = "Vowel Spectral Densities"
  ) +
  theme(
    legend.position = "none",
    panel.spacing = unit(0.1, "lines"),
    axis.ticks.x = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),  # White background # nolint: line_length_linter.
    panel.background = element_rect(fill = "white", color = NA)  # White panel
  )
  scale_color_viridis_d() # nolint: indentation_linter.
  scale_fill_viridis_d()

#print(p1)

#ggsave("vowels_density_single.png", p1, width = 15, height = 9, bg = "white")

p2 <- ggplot(data = data_long, aes(x = freq, y = spectral_density, group = vowel, fill = vowel)) + # nolint
  geom_area(alpha = 0.6) +
  geom_line() +
  theme_ipsum() +
  facet_wrap(~vowel) +
  labs(
    x = "Frequency (Hz)",
    y = "Spectral Density",
    title = "Vowel Spectral Densities"
  ) +
  theme(
    legend.position = "none",
    panel.spacing = unit(0.1, "lines"),
    axis.ticks.x = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),  # White background # nolint: line_length_linter.
    panel.background = element_rect(fill = "white", color = NA)  # White panel
  )
  scale_color_viridis_d() # nolint: indentation_linter.
  scale_fill_viridis_d()

#print(p2)

#ggsave("vowels_density_small_multiple.png", p2, width = 15, height = 9, bg = "white") # nolint

#Aggregate vowel density plot for all speakers

vowel_stats <- data_densities %>%
  group_by(freq, vowel) %>%
  summarise(
    mean_density = mean(density),
    sd_density = sd(density),
    ci_upper = mean_density + 1.96 * sd_density,
    ci_lower = mean_density - 1.96 * sd_density,
    .groups = 'drop'
  )
vowel_stats <- vowel_stats %>%
  mutate(
    ci_upper = pmin(ci_upper, 1),
    ci_lower = pmax(ci_lower, 0)
  )
# Create plot with mean and confidence intervals
p3 <- ggplot(vowel_stats, aes(x = freq, y = mean_density, color = vowel, fill = vowel)) +
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.3, color = NA) +
  geom_line(linewidth = 1) +
  theme_ipsum() +
  labs(
    x = "Frequency (Hz)",
    y = "Spectral Density",
    title = "Vowel Spectral Densities (Mean ± 95% CI)"
  ) +
  theme(
    legend.position = "right",
    panel.spacing = unit(0.1, "lines"),
    axis.ticks.x = element_blank(),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  scale_color_viridis_d() +
  scale_fill_viridis_d()
  scale_y_continuous(limits = c(0, 1))

#print(p3)

#ggsave("vowels_density_aggregate.png", p3, width = 15, height = 9, bg = "white")


#Ridgeline plot aggregate vowels over speakers- Basic
annot <- vowel_stats %>%
  group_by(vowel) %>%
  summarise(
    freq = max(freq)-200,  # Position near the end of frequency axis
    mean_density = max(mean_density),  # Center vertically within each ridge
    label = first(vowel)  # Vowel label
  )

p4 <- ggplot(vowel_stats, aes(x = freq, y = vowel, height = mean_density, fill = vowel)) +
  geom_density_ridges(
    stat = "identity",
    scale = 4,
    alpha = 0.6,
    rel_min_height = 0.01) + 
  geom_text(
    data = annot,
    aes(x = freq, y = vowel, label = label, color = vowel),
    hjust = 0.5,
    vjust = -1,
    fontface = "bold",
    size = 10,
    inherit.aes = FALSE
  ) +
  scale_y_discrete(
    expand = c(0, 0), 
    limits = rev(levels(as.factor(vowel_stats$vowel))),
    labels = NULL  # Remove vowel labels from y-axis
  ) +
  scale_x_continuous(
    expand = c(0, 0),
    breaks = seq(0, max(vowel_stats$freq), by = 500),  # Major grid lines every 1000 Hz
    minor_breaks = seq(0, max(vowel_stats$freq), by = 100)  # Minor grid lines every 500 Hz
  ) +  
  coord_cartesian(clip = "off") +
  theme_ridges() +
  labs(
    x = "Frequency (Hz)",
    y = "Spectral Density",
    title = "Vowel Spectral Densities (Mean ± 95% CI)"
  ) +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_line(color = "grey80", size = 0.5),
    panel.grid.major.y = element_line(color = "grey80", size = 0.25),  # Remove horizontal grid lines
    panel.grid.minor.x = element_line(color = "grey80", size = 0.5),  # Add minor grid lines
    axis.title.x = element_text(hjust = 0.5, margin = margin(t = 10)),  # Center x-axis label and add margin
    axis.title.y = element_text(hjust = 0.5, margin = margin(r = 10)),  # Center y-axis label and add margin
    # Add y-axis scale for spectral density values
    axis.text.y = element_blank()
  ) +
  scale_fill_viridis_d(direction = -1) +
  scale_color_viridis_d(direction = -1)

print(p4)
ggsave("generative_model/spectral_densities/vowels_density_ridges.png", p4, width = 15, height = 9, bg = "white")

#Ridgeline plot aggregate vowels over speakers- Gradient
p5 <- ggplot(vowel_stats, aes(x = freq, y = vowel, height = mean_density, fill = vowel)) +
  geom_density_ridges_gradient(
    stat = "identity",
    scale = 4,
    alpha = 0.6,
    rel_min_height = 0.01) + 
  geom_text(
    data = annot,
    aes(x = freq, y = vowel, label = label, color = vowel),
    hjust = 0.5,
    vjust = 0,  # Center text vertically
    size = 8,
    fontface = "bold",
    inherit.aes = FALSE
  ) +
  scale_y_discrete(expand = c(0, 0), limits = rev(levels(as.factor(vowel_stats$vowel)))) +     # will generally have to set the `expand` option
  scale_x_continuous(
    expand = c(0, 0),
    breaks = seq(0, max(vowel_stats$freq), by = 1000),  # Major grid lines every 1000 Hz
    minor_breaks = seq(0, max(vowel_stats$freq), by = 500)  # Minor grid lines every 500 Hz
  ) +  
  coord_cartesian(clip = "off") +
  theme_ridges() +
  labs(
    x = "Frequency (Hz)",
    y = "Spectral Density",
    title = "Vowel Spectral Densities (Mean ± 95% CI)"
  ) +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_line(color = "grey92", size = 0.5),
    panel.grid.major.y = element_line(color = "grey96", size = 0.25),  # Remove horizontal grid lines
    panel.grid.minor.x = element_line(color = "grey92", size = 0.5)  # Add minor grid lines
  ) +
  scale_fill_viridis_d(direction = -1) +
  scale_color_viridis_d(direction = -1)

print(p5)
ggsave("vowels_density_ridges_gradient.png", p5, width = 15, height = 9, bg = "white")

# Ridgeline gradient for single speaker
annot <- data_long %>%
  group_by(vowel) %>%
  summarise(
    freq = max(freq) - 3000,  # Position near the end of frequency axis
    spectral_density = max(spectral_density),  # Center vertically within each ridge
    label = first(vowel)  # Vowel label
  )
      
p6 <- ggplot(data_long, aes(x = freq, y = vowel, height = spectral_density, fill = vowel)) +
  geom_density_ridges_gradient(
    stat = "identity",
    scale = 4,
    alpha = 0.6,
    rel_min_height = 0.01
  ) + 
  geom_text(
    data = annot,
    aes(x = freq, y = vowel, label = label, color = vowel),
    hjust = 0.5,
    vjust = 0,  # Center text vertically
    size = 8,
    fontface = "bold",
    inherit.aes = FALSE
  ) +
  scale_y_discrete(expand = c(0, 0) , limits = rev(levels(as.factor(data_long$vowel))), labels = NULL) + # nolint: line_length_linter.
  scale_x_continuous(
    expand = c(0, 0),
    breaks = seq(0, max(data_long$freq), by = 1000),  # Major grid lines every 1000 Hz
    minor_breaks = seq(0, max(data_long$freq), by = 500)  # Minor grid lines every 500 Hz
  ) +
  coord_cartesian(clip = "off") +
  theme_ridges() +
  labs(
    x = "Frequency (Hz)",
    y = NULL,
    title = "Vowel Spectral Densities (Single Speaker)"
  ) +
  theme(
    legend.position = "none",
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    axis.text.y = element_blank(),
    panel.grid.major.x = element_line(color = "grey92", size = 0.5),
    panel.grid.major.y = element_line(color = "grey96", size = 0.25),  # Remove horizontal grid lines
    panel.grid.minor.x = element_line(color = "grey92", size = 0.5)  # Add minor grid lines
  ) +
  scale_fill_viridis_d(direction = -1) +
  scale_color_viridis_d(direction = -1)

print(p6)
ggsave("vowels_density_ridges_single.png", p6, width = 15, height = 9, bg = "white") # nolint: line_length_linter.