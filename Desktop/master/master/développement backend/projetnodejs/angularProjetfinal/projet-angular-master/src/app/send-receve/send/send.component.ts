import { Router } from '@angular/router';
import { OnInit } from '@angular/core';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Service } from 'src/app/node.service';
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Pays } from 'src/app/class/pays';
import Swal from 'sweetalert2';
import { findonePays_Devices } from 'src/app/class/PaysDeviseFindOne';

@Component({
  selector: 'app-send',
  templateUrl: './send.component.html',
  styleUrls: ['./send.component.css']
})
export class SendComponent implements OnInit {
  idEmetteur: any;
  idRecepteur: any;
constructor(private router:Router,private http: HttpClient){
 if(sessionStorage.getItem('isloggin')!='true'){
  sessionStorage.setItem('url','send')
  this.router.navigate(['login'])
 }
 }
 verif:string="false"
 allPays:Pays[]|any
 infoem:findonePays_Devices|any
 inforec:findonePays_Devices|any
 
 httpOptions = {
  headers: new HttpHeaders({
    "Content-Type": "application/json"
  })
};
  ngOnInit(): void {

    console.log(sessionStorage.getItem('isloggin'))
    this.http
    .get<Pays[]|any>(
      "http://localhost:3000/api/findAllPays",
      )
      .subscribe(res=>{
this.allPays=res.data
console.log(this.allPays[0])
      })
   
   
  }
  
  submit (form: NgForm) {
     //recuperation des informations de l'emmeteur
    var prenomEm = form.value.prenomemetteur;
    var nomEm=form.value.nomemetteur;
    var cniEm=form.value.cniemetteur
    var phoneEm=form.value.phoneemetteur
//find or create emetteur(client)
    this.http
      .post<any>(
        "http://localhost:3000/api/InsertClient",
        { id: cniEm, nom_client: nomEm,prenom_client:prenomEm,phone:phoneEm },
        this.httpOptions
      )
      .subscribe(res=>{
        this.idEmetteur=res.id
      })
      //recupération information recepteur
      
      var prenom = form.value.prenomrecepteur;
    var nom=form.value.nomrecepteur;
    var cni=form.value.cnirecepteur;
    var phone=form.value.phonerecepteur;
    var montantenvoye=+form.value.montantenvoie;
    var montantreçu=montantenvoye-montantenvoye*0.01
    var idPaysemetteur=form.value.idpaysemetteur
    var idPaysRecepteur=form.value.idpaysrecepteur
    console.log('Emetteur'+idPaysemetteur)
    console.log('Recepteur'+idPaysRecepteur)
    //find or create recepteur(client)
    this.http
      .post<any>(
        "http://localhost:3000/api/InsertClient",
        { id: cni, nom_client: nom,prenom_client:prenom,phone:phone },
        this.httpOptions
      )
      .subscribe(res=>{
       this.idRecepteur=res.id
      })

      //information Pays et device emetteur et recepteur
      this.http
    .post<any>(
      "http://localhost:3000/api/findonePays_Devices",{id:idPaysemetteur},
      this.httpOptions
      )
      .subscribe(res=>{
        
           this.infoem=res.data
          console.log(this.infoem)
      })
      this.http
      .post<any>(
        "http://localhost:3000/api/findonePays_Devices",{id:idPaysRecepteur},
        this.httpOptions
        )
        .subscribe(res=>{
          
            var inforec=res.data
            console.log(inforec
              )
        })
      const swalWithBootstrapButtons = Swal.mixin({
        customClass: {
          confirmButton: 'btn btn-success',
          cancelButton: 'btn btn-danger'
        },
        buttonsStyling: false
      })
      
      swalWithBootstrapButtons.fire({
        title: 'Vous Voulez envoyer à '+prenom+' '+nom+' avec comme CNI '+cni+'Somme envoyé: '+montantenvoye+' montant reçu '+montantreçu+'',
        text: "Vous pouvez annulez d'ici 40 secondes sinon la transaction est confirmé",
        icon: 'warning',
        timer:40000,
        showCancelButton: true,
        confirmButtonText: 'confirmer la transaction',
        cancelButtonText: 'No,Annuler',
        reverseButtons: true
      }).then((result) => {
        if (result.dismiss === Swal.DismissReason.timer) {
          this.router.navigate(['/'])
        }
        if (result.isConfirmed) {
          this.router.navigate(['/'])
        } else if (
          /* Read more about handling dismissals below */
          result.dismiss === Swal.DismissReason.cancel
        ) {
          swalWithBootstrapButtons.fire(
            'Transaction annuler<br> <a href=/>Accueil</a>',
            'Vous Pouvez faire une nouvelle Transaction',
            'error'
          )
        }
      })
      
   }
   calculFrais(event:MouseEvent){
    const index : number=+(event.target as HTMLInputElement).value;
    
   
  
  };
}
