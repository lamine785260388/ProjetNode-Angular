import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Router } from '@angular/router';
import { SousAgence } from 'src/app/class/sous-agence';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-createuser',
  templateUrl: './createuser.component.html',
  styleUrls: ['./createuser.component.css']
})
export class CreateuserComponent implements OnInit {
  constructor(private router:Router,private http: HttpClient){

  }
  donneAgence:SousAgence[]|any
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };
  ngOnInit(): void {
    ///api/findAllSousagence
    this.http
    .get<SousAgence[]|any>(
      "http://localhost:3000/api/findAllSousagence",
      )
      .subscribe(res=>{
        this.donneAgence=res.data
        console.log(this.donneAgence[0].id)
//this.allPays=res.data
//console.log(this.allPays[0])
      })
  }

  submit (form: NgForm) {
    //username password idSousAgence
    let username=form.value.username
    let password=form.value.password
    let idSousAgence=form.value.idSousAgence
    console.log("suis la"+idSousAgence)
    this.http
    .post<any>(
      "http://localhost:3000/api/Inscrire",
      { username: username, password: password,SOUSAGENCEId:idSousAgence },
      this.httpOptions
    )

    .subscribe((res) =>{
      if(res.erreur==false){
        Swal.fire(
          'Inscription Passé!',
          'Avec succés',
          'success'
        )
      }

    
  }
    )}
}
