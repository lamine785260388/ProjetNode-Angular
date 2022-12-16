import { Component } from '@angular/core';
import { OnInit } from '@angular/core';
import { Donne } from './../class/Donne';

import { NgForm } from '@angular/forms';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Router } from '@angular/router';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {
  admin:boolean
  ngOnInit(): void {
    if(sessionStorage.getItem('isAdmin')){
      this.admin=true
    }
    else{
      this.admin=false
    }
  }
 


}
