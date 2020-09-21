table <- read.csv("training.log")
PSNR <- c(t(10.0*log10(1.0/table["loss"])))
plot(1:length(PSNR), PSNR, type="l",ylab="PSNR",xlab="Iterations")

