#!/usr/bin/perl

open(MYFILE, 'CommViolPredUnnormalizedData.txt');
open(OUT, ">commviol.out");
open(IN, ">commviol.in");


while(<MYFILE>){
    chomp;
    s/\?/-1/g; # processes missing variables
    s/\r//g;
    
    @words = split(",", $_);
    
    $N = $#words - 4;
    
#    print "$#words \n";
    
    for my $i (2 .. $N){
	print IN "@words[$i] ";
    }
    print IN "\n";
    
    $M = $N + 4;
    $N = $N + 1;
    
    for my $i ($N .. $M){
	print OUT "@words[$i] ";
    }
    print OUT "\n";
}


close(MYFILE);
close(OUT);
close(IN);
