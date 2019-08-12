#!/bin/bash
echo "Compressing local logs"
tar -cf logs.tar.gz logs/
mkdir ~/trash
echo "Moving logs to trash"
mv logs.tar.gz ~/trash
echo "Cleaning up.."
rm -f logs/*
rm -f debug/
rm -f jobs/*
rm *.std.{out, err}
